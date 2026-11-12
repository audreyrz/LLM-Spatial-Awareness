#!/usr/bin/env python3
"""Compare base vs LoRA-adapted model on Task A / Task B under context ablations.

Conditions
  context   - own narrative
  nocontext - narrative removed (measures phrase priors / language priors)
  shuffled  - narrative from a different example in the same split (same style
              and length, wrong place in the book)

Metrics
  Task A: exact match, normalized match (lowercase, no punctuation), token F1
  Task B: accuracy over A/B/C/D (chance = 0.25), scored from next-token logits
          restricted to the four letters; predicted-letter distribution is
          reported to expose position bias.

Everything is bootstrapped (95% CI over examples). The key quantity is the
interaction:
   (LoRA_ctx - LoRA_ablated) - (base_ctx - base_ablated)
i.e. does adaptation increase how much the model *uses* context, beyond simply
learning the book's phrase distribution?
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONDITIONS, LETTERS, build_prompt_text, normalize_phrase, pick_device_dtype, read_jsonl, token_f1


def clean_pred(s: str) -> str:
    s = s.strip().split("\n")[0].strip()
    return s.strip("\"'`“”‘’ ")


def make_donors(examples):
    """Shuffled-context donor: a far-away example from the SAME text (same
    characters/style, wrong place). Falls back to another text if a text has
    only one example."""
    n = len(examples)
    groups = {}
    for i, e in enumerate(examples):
        groups.setdefault(e.get("book"), []).append(i)
    donors = [None] * n
    for idxs in groups.values():
        m = len(idxs)
        for pos, i in enumerate(idxs):
            if m > 1:
                donors[i] = examples[idxs[(pos + max(1, m // 2)) % m]]
            else:
                donors[i] = examples[(i + max(1, n // 2)) % n]
    return donors


@torch.no_grad()
def run_task_a(model, tok, examples, donors, condition, bsz, max_new, device):
    preds = []
    for i in range(0, len(examples), bsz):
        exs = examples[i : i + bsz]
        prompts = [build_prompt_text(tok, e, condition, donors[i + j]) for j, e in enumerate(exs)]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        preds += [clean_pred(t) for t in texts]
    em = np.array([float(p.strip() == e["target"].strip()) for p, e in zip(preds, examples)])
    nm = np.array([float(normalize_phrase(p) == normalize_phrase(e["target"])) for p, e in zip(preds, examples)])
    f1 = np.array([token_f1(p, e["target"]) for p, e in zip(preds, examples)])
    return {"exact_match": em, "normalized_match": nm, "token_f1": f1}, preds


@torch.no_grad()
def run_task_b(model, tok, examples, donors, condition, bsz, device):
    letter_ids = []
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"letter {L} is not a single token for this tokenizer"
        letter_ids.append(ids[0])
    preds = []
    for i in range(0, len(examples), bsz):
        exs = examples[i : i + bsz]
        prompts = [build_prompt_text(tok, e, condition, donors[i + j]) for j, e in enumerate(exs)]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, -1, :][:, letter_ids]  # left padding -> last position
        preds += [LETTERS[k] for k in logits.argmax(-1).tolist()]
    acc = np.array([float(p == e["label"]) for p, e in zip(preds, examples)])
    return {"accuracy": acc}, preds


def boot_ci(x, rng, n_boot):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(scores, rng, n_boot):
    return {
        k: {"mean": float(v.mean()), "ci95": boot_ci(v, rng, n_boot)} for k, v in scores.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="runs/pilot/best")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tasks", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N examples per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--pred_out", default="predictions.jsonl")
    args = ap.parse_args()
    if "context" not in args.conditions:
        args.conditions = ["context"] + args.conditions

    rng = np.random.default_rng(args.seed)
    device, dtype = pick_device_dtype()
    if device == "mps" and args.bsz > 1:
        print(f"MPS detected; reducing evaluation batch size from {args.bsz} to 1 to limit memory use")
        args.bsz = 1
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = {}
    for t in args.tasks:
        rows = read_jsonl(Path(args.data_dir) / f"task_{t.lower()}_{args.split}.jsonl")
        data[t] = rows[: args.limit] if args.limit else rows
    donors = {t: make_donors(rows) for t, rows in data.items()}

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    results, raw_scores, pred_rows = {}, {}, []

    def evaluate_model(name, model):
        results[name], raw_scores[name] = {}, {}
        for cond in args.conditions:
            results[name][cond], raw_scores[name][cond] = {}, {}
            for t in args.tasks:
                ex = data[t]
                if t == "A":
                    sc, preds = run_task_a(model, tok, ex, donors[t], cond, args.bsz, args.max_new_tokens, device)
                else:
                    sc, preds = run_task_b(model, tok, ex, donors[t], cond, args.bsz, device)
                entry = {"n": len(ex), **summarize(sc, rng, args.n_boot)}
                hk = "normalized_match" if t == "A" else "accuracy"
                entry["per_text"] = {
                    bk: {"n": len(ii), "mean": float(sc[hk][ii].mean())}
                    for bk in sorted({e["book"] for e in ex})
                    for ii in [[k for k, e in enumerate(ex) if e["book"] == bk]]
                }
                if t == "B":
                    entry["chance"] = 0.25
                    entry["pred_letter_distribution"] = {L: preds.count(L) for L in LETTERS}
                results[name][cond][f"task_{t.lower()}"] = entry
                raw_scores[name][cond][f"task_{t.lower()}"] = sc
                for e, p in zip(ex, preds):
                    pred_rows.append({"model": name, "condition": cond, "task": t, "id": e["id"], "pred": p,
                                      "gold": e["target"] if t == "A" else e["label"]})
                headline = entry.get("normalized_match", entry.get("accuracy"))
                print(f"[{name:4s}] {cond:9s} task_{t.lower()}  n={len(ex)}  {headline['mean']:.3f}  CI {headline['ci95']}")

    evaluate_model("base", base)

    from peft import PeftModel

    adapted = PeftModel.from_pretrained(base, args.adapter).eval()
    evaluate_model("lora", adapted)

    # ---- contrasts (paired bootstrap over examples) -------------------------
    metric_keys = []
    if "A" in args.tasks:
        metric_keys += [("task_a", "exact_match"), ("task_a", "normalized_match"), ("task_a", "token_f1")]
    if "B" in args.tasks:
        metric_keys += [("task_b", "accuracy")]

    def paired(fn, task, metric):
        arrs = {(m, c): raw_scores[m][c][task][metric] for m in ("base", "lora") for c in args.conditions}
        n = len(next(iter(arrs.values())))
        point = fn(lambda k: arrs[k].mean())
        idx = rng.integers(0, n, size=(args.n_boot, n))
        boots = [fn(lambda k, ii=ii: arrs[k][ii].mean()) for ii in idx]
        return {"estimate": float(point), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

    summary = {}
    for task, metric in metric_keys:
        key = f"{task}.{metric}"
        summary[key] = {
            "context_gain_base": paired(lambda g: g(("base", "context")) - g(("base", "nocontext")), task, metric)
            if "nocontext" in args.conditions else None,
            "context_gain_lora": paired(lambda g: g(("lora", "context")) - g(("lora", "nocontext")), task, metric)
            if "nocontext" in args.conditions else None,
            "adaptation_effect_context": paired(lambda g: g(("lora", "context")) - g(("base", "context")), task, metric),
        }
        for ab in [c for c in args.conditions if c != "context"]:
            summary[key][f"interaction_vs_{ab}"] = paired(
                lambda g, ab=ab: (g(("lora", "context")) - g(("lora", ab))) - (g(("base", "context")) - g(("base", ab))),
                task, metric,
            )
            summary[key][f"adaptation_effect_{ab}"] = paired(
                lambda g, ab=ab: g(("lora", ab)) - g(("base", ab)), task, metric
            )

    out = {
        "meta": {
            "model": args.model,
            "adapter": args.adapter,
            "split": args.split,
            "n_examples": {t: len(r) for t, r in data.items()},
            "note": "Split by whole text. Task B = next ANNOTATED event, not necessarily next physical movement. Example-level CIs ignore within-text correlation; see per_text.",
            "args": vars(args),
        },
        "results": results,
        "contrasts": summary,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    with open(args.pred_out, "w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} and {args.pred_out}")


if __name__ == "__main__":
    main()
