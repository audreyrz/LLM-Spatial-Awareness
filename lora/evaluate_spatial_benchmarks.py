#!/usr/bin/env python3
"""Compare a base causal LM and its PEFT adapter on StepGame and SpaRP.

These are external static spatial-relation QA benchmarks, not narrative
movement benchmarks. Data is loaded from the official Hugging Face datasets.
SpaRP test items include verified reasoning paths; this script deliberately
does not expose the `reasoning` field to the model.

Examples (run from lora/):
  python evaluate_spatial_benchmarks.py --benchmark stepgame --limit 5000
  python evaluate_spatial_benchmarks.py --benchmark sparp \
      --config "SpaRP-PS1 (SpaRTUN)"

Requires the optional `datasets` package in addition to requirements.txt.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import pick_device_dtype


STEPGAME_RELATIONS = [
    "upper-left", "upper-right", "lower-left", "lower-right",
    "left", "right", "above", "below", "overlap",
]


def load_examples(args):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing optional dependency; install with: python -m pip install datasets") from exc
    if args.benchmark == "stepgame":
        ds = load_dataset("ZhengyanShi/StepGame", split=args.split)
        rows = []
        for i, ex in enumerate(ds):
            rows.append({
                "id": str(i),
                "context": "\n".join(ex["story"]),
                "question": ex["question"],
                "targets": [str(ex["label"])],
                "choices": STEPGAME_RELATIONS,
                "group": str(ex.get("k_hop", "unknown")),
            })
    else:
        ds = load_dataset("UKPLab/sparp", args.config, split=args.split)
        rows = []
        for i, ex in enumerate(ds):
            # SpaRP contains both StepGame and SpaRTUN sources. `source_data`
            # is retained in the prediction output for disaggregated analysis.
            rows.append({
                "id": f"{ex.get('context_id', '')}-{ex.get('question_id', i)}",
                "context": ex["context"],
                "question": ex["question"],
                "targets": list(ex["targets"]),
                "choices": list(ex["target_choices"]),
                "group": str(ex.get("num_hop", "unknown")),
                "source": ex.get("source_data", "unknown"),
            })
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("The selected benchmark split is empty")
    # Keep the benchmark in a consistent four-option format. Retain every gold
    # relation, sample the remaining options from the example's candidate pool,
    # and randomize their displayed order deterministically by seed and id.
    narrowed = []
    for ex in rows:
        targets = list(dict.fromkeys(map(normalize_relation, ex["targets"])))
        if len(targets) > 4:
            raise SystemExit(f"Example {ex['id']} has more than four gold relations; cannot form a four-choice item")
        pool = list(dict.fromkeys(map(normalize_relation, ex["choices"])))
        negatives = [choice for choice in pool if choice not in targets]
        if len(negatives) < 4 - len(targets):
            raise SystemExit(f"Example {ex['id']} has too few distractor labels to form four choices")
        local_rng = random.Random(f"{args.seed}:{ex['id']}")
        choices = targets + local_rng.sample(negatives, 4 - len(targets))
        local_rng.shuffle(choices)
        ex["targets"] = targets
        ex["choices"] = choices
        narrowed.append(ex)
    rows = narrowed
    return rows


def prompt_text(tokenizer, ex):
    choices = ", ".join(ex["choices"])
    user = (
        "Answer the spatial reasoning question using only the context. "
        "Choose every relation that is supported. Return only the relation "
        "label or labels, separated by commas; do not explain.\n\n"
        f"Context:\n{ex['context']}\n\n"
        f"Question: {ex['question']}\n"
        f"Possible relation labels: {choices}\n"
        "Answer:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_relation(label):
    value = re.sub(r"\s+", " ", str(label).strip().lower())
    value = value.replace("–", "-").replace("—", "-").replace(" ", "-")
    if value in {"overlapping", "partially-overlapping"}:
        return "overlap" if value == "overlapping" else value
    return value


def extract_relations(text, choices):
    """Extract complete canonical labels, preferring longer multiword labels."""
    text = text.lower().replace("–", "-").replace("—", "-")
    found = set()
    # Normalize both source text and labels so punctuation variants match.
    searchable = re.sub(r"[^a-z0-9]+", " ", text)
    for choice in sorted(choices, key=len, reverse=True):
        norm = normalize_relation(choice).replace("-", " ")
        pattern = r"(?<![a-z0-9])" + re.escape(norm) + r"(?![a-z0-9])"
        if re.search(pattern, searchable):
            found.add(normalize_relation(choice))
    return found


def render_prompts(tokenizer, examples):
    return [prompt_text(tokenizer, ex) for ex in examples]


@torch.inference_mode()
def predict(model, tokenizer, examples, batch_size, max_new_tokens, device):
    model.eval()
    prompts = render_prompts(tokenizer, examples)
    outputs = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start:start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # With left padding, all rows have the same input length, so the slice
        # begins at the first generated token for every row.
        outputs.extend(tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ))
    return [text.strip() for text in outputs]


def summarize(examples, predictions, n_boot, seed):
    gold_sets = [set(map(normalize_relation, ex["targets"])) for ex in examples]
    pred_sets = [extract_relations(pred, ex["choices"]) for pred, ex in zip(predictions, examples)]
    exact = np.asarray([float(g == p) for g, p in zip(gold_sets, pred_sets)])
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(exact), size=(n_boot, len(exact)))
    acc_ci = np.percentile(exact[boot].mean(axis=1), [2.5, 97.5]).tolist()

    labels = sorted(set().union(*gold_sets, *pred_sets))
    per_relation = {}
    class_f1s = []
    for label in labels:
        ctp = sum(label in g and label in p for g, p in zip(gold_sets, pred_sets))
        cfp = sum(label not in g and label in p for g, p in zip(gold_sets, pred_sets))
        cfn = sum(label in g and label not in p for g, p in zip(gold_sets, pred_sets))
        p = ctp / (ctp + cfp) if ctp + cfp else 0.0
        r = ctp / (ctp + cfn) if ctp + cfn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_relation[label] = {"precision": p, "recall": r, "f1": f1}
        class_f1s.append(f1)
    return {
        "n": len(examples),
        "exact_match_accuracy": float(exact.mean()),
        "exact_match_ci95": [float(x) for x in acc_ci],
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_relation_f1": float(np.mean(class_f1s)) if class_f1s else 0.0,
        "per_relation": per_relation,
        "per_group_exact_match": {
            group: float(np.mean([exact[i] for i, ex in enumerate(examples) if ex["group"] == group]))
            for group in sorted({ex["group"] for ex in examples})
        },
    }, pred_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["stepgame", "sparp"], required=True)
    parser.add_argument("--config", default="SpaRP-PS1 (SpaRTUN)",
                        help="SpaRP HF config; ignored for StepGame")
    parser.add_argument("--split", default="test", help="HF split name (test by default)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="runs/pilot/best")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None, help="debug/smoke-run cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--pred_out", default=None)
    args = parser.parse_args()

    examples = load_examples(args)
    device, dtype = pick_device_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    results, pred_rows, exact_scores = {}, [], {}
    for name, model in (("base", base),):
        text_predictions = predict(model, tokenizer, examples, args.batch_size,
                                   args.max_new_tokens, device)
        summary, pred_sets = summarize(examples, text_predictions, args.n_boot, args.seed)
        results[name] = summary
        exact_scores[name] = np.asarray([
            set(map(normalize_relation, ex["targets"])) == pred_set
            for ex, pred_set in zip(examples, pred_sets)
        ], dtype=float)
        for ex, raw, pred_set in zip(examples, text_predictions, pred_sets):
            pred_rows.append({
                "model": name, "id": ex["id"], "group": ex["group"],
                "source": ex.get("source"), "gold": ex["targets"], "choices": ex["choices"],
                "pred": sorted(pred_set), "raw_output": raw,
            })
        print(f"[{name:5s}] exact={summary['exact_match_accuracy']:.3f} "
              f"micro_f1={summary['micro_f1']:.3f} n={summary['n']}")

    adapted = PeftModel.from_pretrained(base, args.adapter).eval()
    text_predictions = predict(adapted, tokenizer, examples, args.batch_size,
                               args.max_new_tokens, device)
    summary, pred_sets = summarize(examples, text_predictions, args.n_boot, args.seed)
    results["finetuned"] = summary
    exact_scores["finetuned"] = np.asarray([
        set(map(normalize_relation, ex["targets"])) == pred_set
        for ex, pred_set in zip(examples, pred_sets)
    ], dtype=float)
    for ex, raw, pred_set in zip(examples, text_predictions, pred_sets):
        pred_rows.append({
            "model": "finetuned", "id": ex["id"], "group": ex["group"],
            "source": ex.get("source"), "gold": ex["targets"], "choices": ex["choices"],
            "pred": sorted(pred_set), "raw_output": raw,
        })
    print(f"[finetuned] exact={summary['exact_match_accuracy']:.3f} "
          f"micro_f1={summary['micro_f1']:.3f} n={summary['n']}")

    rng = np.random.default_rng(args.seed)
    paired_delta = exact_scores["finetuned"] - exact_scores["base"]
    boot_idx = rng.integers(0, len(paired_delta), size=(args.n_boot, len(paired_delta)))
    delta_ci = np.percentile(paired_delta[boot_idx].mean(axis=1), [2.5, 97.5]).tolist()

    default_stem = f"{args.benchmark}_{args.config.replace(' ', '_').replace('/', '-') if args.benchmark == 'sparp' else 'official'}"
    out_path = Path(args.out or f"{default_stem}_results.json")
    pred_path = Path(args.pred_out or f"{default_stem}_predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "benchmark": args.benchmark,
            "dataset": "ZhengyanShi/StepGame" if args.benchmark == "stepgame" else "UKPLab/sparp",
            "config": args.config if args.benchmark == "sparp" else None,
            "split": args.split,
            "choices_per_example": 4,
            "choice_order_seed": args.seed,
            "model": args.model,
            "adapter": args.adapter,
            "n_examples": len(examples),
            "note": "StepGame and SpaRP measure static relative-position question answering, not narrative movement/change. SpaRP reasoning text is excluded from model inputs.",
            "args": vars(args),
        },
        "results": results,
        "paired_comparison": {
            "metric": "exact_match_accuracy",
            "finetuned_minus_base": float(paired_delta.mean()),
            "ci95": [float(x) for x in delta_ci],
        },
    }, indent=2), encoding="utf-8")
    with pred_path.open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} and {pred_path}")


if __name__ == "__main__":
    main()
