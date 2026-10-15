#!/usr/bin/env python3
"""LoRA-adapt a small instruct LM on Task A + Task B (train split only).

* Loss is computed on the answer tokens only (phrase for A, letter for B).
* Training always uses the *context* condition; the no-context / shuffled
  conditions are evaluation-time controls (see evaluate.py).
* Per-example mean loss, so the 1-token Task B answers are not drowned out by
  multi-token Task A answers.
* Best adapter is chosen by validation loss (val split, never test).
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common import answer_text, build_prompt_text, pick_device_dtype, read_jsonl


def encode(tok, ex, max_len):
    p = tok(build_prompt_text(tok, ex, "context"), add_special_tokens=False)["input_ids"]
    a = tok(answer_text(ex) + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:
        return None
    return {
        "task": ex["task"],
        "input_ids": p + a,
        "labels": [-100] * len(p) + a,
    }


def collate(batch, pad_id, device):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return ids.to(device), lab.to(device), att.to(device)


def per_example_loss(model, ids, lab, att, use_amp, device):
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)
    with ctx:
        logits = model(input_ids=ids, attention_mask=att).logits
    logits = logits[:, :-1].float()
    tgt = lab[:, 1:]
    tok_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none"
    ).view(tgt.shape)
    mask = (tgt != -100).float()
    return (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)


@torch.no_grad()
def evaluate_loss(model, data, pad_id, bsz, use_amp, device):
    model.eval()
    sums = {"A": [0.0, 0], "B": [0.0, 0]}
    for i in range(0, len(data), bsz):
        batch = data[i : i + bsz]
        ids, lab, att = collate(batch, pad_id, device)
        losses = per_example_loss(model, ids, lab, att, use_amp, device)
        for b, l in zip(batch, losses.tolist()):
            sums[b["task"]][0] += l
            sums[b["task"]][1] += 1
    out = {t: (s / n if n else float("nan")) for t, (s, n) in sums.items()}
    vals = [v for v in out.values() if not math.isnan(v)]
    out["mean"] = sum(vals) / len(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out_dir", default="runs/pilot")
    ap.add_argument("--tasks", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = pick_device_dtype()
    use_amp = device == "cuda" and dtype == torch.bfloat16
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(split):
        rows = []
        for t in args.tasks:
            rows += read_jsonl(Path(args.data_dir) / f"task_{t.lower()}_{split}.jsonl")
        enc = [encode(tok, r, args.max_len) for r in rows]
        dropped = sum(e is None for e in enc)
        return [e for e in enc if e is not None], dropped

    train, d_tr = load("train")
    val, d_va = load("val")
    print(f"train={len(train)} (dropped {d_tr} too long)  val={len(val)} (dropped {d_va})")
    if not train or not val:
        raise SystemExit("empty train/val set")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / (args.bsz * args.grad_accum))
    total = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(args.warmup_frac * total)), total)

    log = {"args": vars(args), "epochs": []}
    base_val = evaluate_loss(model.eval(), val, tok.pad_token_id, args.bsz, use_amp, device)
    print("val loss at init (adapter is identity):", base_val)
    log["val_loss_init"] = base_val
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        run, n_run = 0.0, 0
        for i in range(0, len(train), args.bsz):
            batch = train[i : i + args.bsz]
            ids, lab, att = collate(batch, tok.pad_token_id, device)
            loss = per_example_loss(model, ids, lab, att, use_amp, device).mean()
            (loss / args.grad_accum).backward()
            run += loss.item()
            n_run += 1
            if n_run % args.grad_accum == 0 or i + args.bsz >= len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        v = evaluate_loss(model, val, tok.pad_token_id, args.bsz, use_amp, device)
        rec = {"epoch": ep, "train_loss": run / max(1, n_run), "val_loss": v}
        log["epochs"].append(rec)
        print(json.dumps(rec))
        if v["mean"] < best:
            best = v["mean"]
            model.save_pretrained(out_dir / "best")
            log["best_epoch"] = ep

    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"best epoch {log.get('best_epoch')}, adapter saved to {out_dir / 'best'}")


if __name__ == "__main__":
    main()
