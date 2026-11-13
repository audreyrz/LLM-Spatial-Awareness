#!/usr/bin/env python3
"""Linear probe for ordered spatial entity-pair representations on StepGame.

Entity vectors are the final-layer contextual states at their mentions in the
question. A linear classifier receives [A, B, |A-B|, A*B] and predicts one of
the nine StepGame relations. The official train/test split is preserved.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import pick_device_dtype
from evaluate_spatial_benchmarks import STEPGAME_RELATIONS, normalize_relation


def pair_entities(question):
    found = re.findall(r"\b[A-Z]\b", question)
    if len(found) < 2:
        raise ValueError(f"Could not identify two entity symbols in question: {question!r}")
    return found[-2], found[-1]


def make_prompt(ex):
    return "Context:\n" + "\n".join(ex["story"]) + "\nQuestion: " + ex["question"]


@torch.inference_mode()
def extract_features(model, tokenizer, rows, device, max_length):
    model.eval()
    features, labels = [], []
    relation_ids = {normalize_relation(r): i for i, r in enumerate(STEPGAME_RELATIONS)}
    for row in rows:
        prompt = make_prompt(row)
        entity_a, entity_b = pair_entities(row["question"])
        offsets = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=max_length, return_offsets_mapping=True)
        spans = []
        for entity in (entity_a, entity_b):
            start = prompt.rfind(entity)
            if start < 0:
                raise ValueError(f"Entity {entity!r} absent from prompt")
            end = start + len(entity)
            positions = [i for i, (a, b) in enumerate(offsets["offset_mapping"][0].tolist())
                         if b > start and a < end]
            if not positions:
                raise ValueError(f"Token span not found for entity {entity!r}")
            spans.append(positions)
        model_inputs = {k: v.to(device) for k, v in offsets.items() if k != "offset_mapping"}
        hidden = model(**model_inputs, output_hidden_states=True).hidden_states[-1][0]
        a = hidden[spans[0]].mean(0)
        b = hidden[spans[1]].mean(0)
        features.append(torch.cat([a, b, torch.abs(a - b), a * b]).float().cpu())
        label = normalize_relation(row["label"])
        if label not in relation_ids:
            raise ValueError(f"Unknown StepGame label {row['label']!r}")
        labels.append(relation_ids[label])
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def fit_probe(train_x, train_y, test_x, test_y, seed, steps, lr):
    torch.manual_seed(seed)
    x_mean = train_x.mean(0, keepdim=True)
    x_std = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_x, test_x = (train_x - x_mean) / x_std, (test_x - x_mean) / x_std
    clf = torch.nn.Linear(train_x.shape[1], len(STEPGAME_RELATIONS))
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(train_x), train_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = clf(test_x).argmax(-1)
        acc = float((pred == test_y).float().mean())
    return {"accuracy": acc, "n_train": len(train_y), "n_test": len(test_y),
            "chance": 1 / len(STEPGAME_RELATIONS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="optional LoRA checkpoint to compare with base")
    ap.add_argument("--dataset", default="ZhengyanShi/StepGame")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_test", type=int, default=None)
    ap.add_argument("--probe_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="spatial_probe_results.json")
    args = ap.parse_args()
    dataset = load_dataset(args.dataset)
    train_rows, test_rows = dataset["train"], dataset["test"]
    if args.limit_train: train_rows = train_rows.select(range(min(args.limit_train, len(train_rows))))
    if args.limit_test: test_rows = test_rows.select(range(min(args.limit_test, len(test_rows))))
    device, dtype = pick_device_dtype()
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    comparisons = [("base", base)]
    if args.adapter:
        from peft import PeftModel
        comparisons.append(("finetuned", PeftModel.from_pretrained(base, args.adapter)))
    result = {"model": args.model, "adapter": args.adapter, "protocol": __doc__, "scores": {}}
    for name, model in comparisons:
        train_x, train_y = extract_features(model, tok, train_rows, device, args.max_length)
        test_x, test_y = extract_features(model, tok, test_rows, device, args.max_length)
        result["scores"][name] = fit_probe(train_x, train_y, test_x, test_y,
                                           args.seed, args.probe_steps, args.lr)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["scores"], indent=2))


if __name__ == "__main__": main()
