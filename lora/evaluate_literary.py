#!/usr/bin/env python3
"""Evaluate a causal LM on generated literary movement MCQs."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import LETTERS, pick_device_dtype, read_jsonl


def prompt(row, demonstrations=(), choices=None):
    choices = choices or row["choices"]
    shots = ""
    if demonstrations:
        shots = "\n\nExamples:\n" + "\n\n".join(
            "Passage: " + d["passage"] + "\nCandidates:\n" + "\n".join(f"{L}. {c}" for L,c in zip(LETTERS,d["choices"])) + f"\nAnswer: {d['answer']}"
            for d in demonstrations
        )
    return (
        "Read the passage and choose the candidate that is an annotated movement span in the narrative. "
        "Reply with one letter only: A, B, C, or D.\n" + shots + "\n\nPassage:\n" + row["passage"] +
        "\n\nCandidates:\n" + "\n".join(f"{L}. {c}" for L,c in zip(LETTERS,choices)) + "\nAnswer:"
    )


def parse_answer(response):
    m = re.match(r"\s*\(?([ABCD])\)?(?:\b|[.)])", response.upper())
    if m: return m.group(1)
    hits = re.findall(r"\b([ABCD])\b", response.upper())
    return hits[0] if len(set(hits)) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="benchmark JSONL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--few_shot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="literary_predictions.jsonl")
    ap.add_argument("--summary", default="literary_results.json")
    args = ap.parse_args()
    rows = read_jsonl(args.data)
    if args.limit: rows = rows[:args.limit]
    demos = read_jsonl(args.data.replace("test", "train"))[:args.few_shot] if args.few_shot else []
    if any(d["id"] in {r["id"] for r in rows} for d in demos): raise ValueError("few-shot examples overlap evaluation rows")
    if args.model == "random":
        model = tok = device = None
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device, dtype = pick_device_dtype()
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).eval()
        if tok.pad_token is None: tok.pad_token = tok.eos_token
    results, summaries = [], defaultdict(lambda: {"n": 0, "correct": 0, "parse_failures": 0, "by_novel": defaultdict(lambda: [0,0])})
    for row in rows:
        # Randomize displayed option order on every run and score against the
        # corresponding remapped answer letter.
        import random
        order = list(range(len(LETTERS)))
        random.Random(f"{args.seed}:{row['id']}").shuffle(order)
        choices = [row["choices"][i] for i in order]
        gold_letter = LETTERS[order.index(row["answer_index"])]
        if args.model == "random":
            pred = random.Random(f"{args.seed}:{row['id']}").choice(LETTERS)
            response = pred
        else:
            text = prompt(row, demos, choices)
            if getattr(tok, "chat_template", None):
                text = tok.apply_chat_template([{"role":"user", "content":text}], tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
            response = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_answer(response)
        correct = pred == gold_letter
        results.append({"id":row["id"], "source_novel":row["source_novel"], "distractor_type":row["distractor_type"],
                        "prediction":pred, "gold":gold_letter, "correct":correct, "response":response,
                        "gold_span":row["gold_span"], "choices":choices})
        s = summaries[row["distractor_type"]]; s["n"] += 1; s["correct"] += int(correct); s["parse_failures"] += int(pred is None)
        s["by_novel"][row["source_novel"]][0] += int(correct); s["by_novel"][row["source_novel"]][1] += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    final = {}
    for key, s in summaries.items():
        final[key] = {"n":s["n"], "accuracy":s["correct"]/s["n"] if s["n"] else None,
                      "chance":0.25, "parse_failures":s["parse_failures"],
                      "by_novel":{n:{"correct":v[0],"n":v[1],"accuracy":v[0]/v[1]} for n,v in s["by_novel"].items()}}
    Path(args.summary).write_text(json.dumps({"model":args.model,"adapter":args.adapter,"few_shot":args.few_shot,
                                              "seed":args.seed,"data":args.data,"results":final},indent=2),encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__": main()
