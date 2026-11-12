#!/usr/bin/env python3
"""LoRA fine-tune Qwen on StepGame spatial-relation QA.

Uses a seeded validation slice from the official train split and leaves the
official test split untouched. Prompt format matches evaluate_spatial_benchmarks.py.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from common import pick_device_dtype
from evaluate_spatial_benchmarks import STEPGAME_RELATIONS, normalize_relation


def format_user(ex):
    choices = ", ".join(STEPGAME_RELATIONS)
    context = "\n".join(ex["story"])
    return (
        "Answer the spatial reasoning question using only the context. Choose every relation "
        "that is supported. Return only the relation label or labels, separated by commas; do not explain.\n\n"
        f"Context:\n{context}\n\nQuestion: {ex['question']}\n"
        f"Possible relation labels: {choices}\nAnswer:"
    )


def answer_for(ex):
    """Convert StepGame's one gold relation into the evaluator's canonical spelling."""
    gold = normalize_relation(ex["label"])
    labels = {normalize_relation(x): x for x in STEPGAME_RELATIONS}
    if gold not in labels:
        raise ValueError(f"Unknown StepGame relation label: {ex['label']!r}")
    return labels[gold]


def encode_row(tokenizer, ex, max_len):
    user = format_user(ex)
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = user + "\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    answer_ids = tokenizer(answer_for(ex) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(answer_ids) >= max_len:
        return None
    prompt_ids = prompt_ids[-(max_len - len(answer_ids)):]
    ids = prompt_ids + answer_ids
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids,
            "attention_mask": [1] * len(ids)}


def batches(data, batch_size, rng):
    order = list(range(len(data)))
    rng.shuffle(order)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def collate(examples, pad_id, device):
    length = max(len(x["input_ids"]) for x in examples)
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), length), dtype=torch.long)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        ids[i, :n] = torch.tensor(ex["input_ids"])
        labels[i, :n] = torch.tensor(ex["labels"])
        mask[i, :n] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def batch_loss(model, batch, pad_id, device):
    ids, labels, mask = collate(batch, pad_id, device)
    outputs = model(input_ids=ids, attention_mask=mask)
    logits = outputs.logits[:, :-1, :].contiguous().float()
    targets = labels[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
    )
    return loss


@torch.no_grad()
def evaluate_loss(model, data, batch_size, pad_id, device):
    model.eval()
    losses = []
    for start in range(0, len(data), batch_size):
        loss = batch_loss(model, data[start:start + batch_size], pad_id, device)
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset", default="ZhengyanShi/StepGame")
    ap.add_argument("--out_dir", default="runs/qwen-stepgame")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--warmup_fraction", type=float, default=0.05)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit_train", type=int, default=None, help="optional debugging cap")
    ap.add_argument("--limit_val", type=int, default=None, help="optional debugging cap")
    args = ap.parse_args()
    if not 0 < args.val_fraction < 0.5:
        ap.error("--val_fraction must be between 0 and 0.5")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # Build training/validation only from StepGame's official train split.
    raw = load_dataset(args.dataset, split="train")
    split = raw.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"  # match evaluation truncation

    def prepare(rows, limit):
        if limit: rows = rows.select(range(min(limit, len(rows))))
        output = [encode_row(tokenizer, row, args.max_len) for row in rows]
        return [x for x in output if x is not None]

    train = prepare(split["train"], args.limit_train)
    val = prepare(split["test"], args.limit_val)
    if not train or not val: raise SystemExit("Training or validation data is empty")

    device, dtype = pick_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    updates_per_epoch = math.ceil(math.ceil(len(train) / args.batch_size) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_fraction), total_updates
    )
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val, history, step = float("inf"), [], 0
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        rng = random.Random(args.seed + epoch)
        epoch_losses, pending = [], 0
        for indices in batches(train, args.batch_size, rng):
            loss = batch_loss(model, [train[i] for i in indices], tokenizer.pad_token_id, device)
            epoch_losses.append(float(loss.detach()))
            (loss / args.grad_accum).backward(); pending += 1
            if pending == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1; pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
        val_loss = evaluate_loss(model, val, args.batch_size, tokenizer.pad_token_id, device)
        record = {"epoch": epoch + 1, "train_loss": sum(epoch_losses)/len(epoch_losses), "val_loss": val_loss}
        history.append(record); print(json.dumps(record))
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
    (out_dir / "train_log.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "train_split": "train",
        "official_test_used_for_training": False, "n_train": len(train), "n_val": len(val),
        "best_val_loss": best_val, "history": history, "args": vars(args),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
