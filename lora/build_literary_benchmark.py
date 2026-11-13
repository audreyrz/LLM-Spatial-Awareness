#!/usr/bin/env python3
"""Build a novel-held-out multiple-choice benchmark for movement spans.

The checkout's source is corpus/annotations/consensus_final_*.json paired with
corpus/texts/*_excerpt.txt. Annotation spans are aligned by exact text when
offsets have drifted, using the same rules as lora/build_data.py.
"""
import argparse
import json
import random
import re
from pathlib import Path

from common import LETTERS, normalize_phrase, write_jsonl

ACTION = re.compile(r"\b(?:walk|run|go|come|move|travel|cross|climb|fall|enter|leave|return|turn|step|rush|race|ride|drive|sail|fly|swim|wander|march|approach|depart|flee|follow|carry|bring|take|reach|head|set|pass|descend|ascend|stumble|creep|crawl|hurry|escape|arrive|proceed)(?:s|ed|ing)?\b", re.I)
SPATIAL = re.compile(r"\b(?:to|from|into|onto|upon|through|across|along|toward|towards|away|out|off|over|under|beneath|behind|between|beyond|up|down|back|near|inside|outside|past|around)\b", re.I)


def sentence_spans(text):
    """Conservative sentence-like spans retaining original offsets."""
    out, start = [], 0
    for m in re.finditer(r"[.!?]+(?:[\"'’”)]*)?(?:\s+|$)|\n+", text):
        end = m.start() if m.group().startswith("\n") else m.end()
        a, b = start, end
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b-1].isspace(): b -= 1
        if b > a: out.append({"start": a, "end": b, "text": text[a:b]})
        start = m.end()
    a, b = start, len(text)
    while a < b and text[a].isspace(): a += 1
    while b > a and text[b-1].isspace(): b -= 1
    if b > a: out.append({"start": a, "end": b, "text": text[a:b]})
    return out


def candidates_for(book, seed, strategy):
    text, events = book["text"], book["events"]
    sentences = sentence_spans(text)
    nonmovement = [s for s in sentences if not any(s["start"] < e["end"] and e["start"] < s["end"] for e in events)]
    unique, seen = [], set()
    for s in nonmovement:
        key = normalize_phrase(s["text"])
        if key and key not in seen:
            seen.add(key); unique.append(s)
    nonmovement = unique
    if strategy == "hard":
        nonmovement.sort(key=lambda s: (-(bool(ACTION.search(s["text"])) + bool(SPATIAL.search(s["text"]))), s["start"]))
    return nonmovement


def make_questions(book, split, seed, strategy):
    rng = random.Random(f"{seed}:{book['name']}:{strategy}")
    distractors = candidates_for(book, seed, strategy)
    rows = []
    for ev in book["events"]:
        gold = ev["text"]
        gnorm = normalize_phrase(gold)
        pool = [s for s in distractors if normalize_phrase(s["text"]) != gnorm]
        if strategy == "random": rng.shuffle(pool)
        # Hard candidates are ranked, but sampled from the highest-ranked pool to avoid a fixed template.
        pool = pool[:max(3, min(len(pool), 20))]
        if len(pool) < 3: continue
        ds = rng.sample(pool, 3)
        choices = [gold] + [d["text"] for d in ds]
        order = list(range(4)); rng.shuffle(order)
        ordered = [choices[i] for i in order]
        rows.append({
            "id": f"{book['name']}-{ev['idx']}-{strategy}", "source_novel": book["name"], "split": split,
            "passage": book["text"][max(0, ev["start"]-1200):min(len(book["text"]), ev["end"]+1200)],
            "passage_start": max(0, ev["start"]-1200), "gold_span": gold, "gold_start": ev["start"], "gold_end": ev["end"],
            "choices": ordered, "answer_index": order.index(0), "answer": LETTERS[order.index(0)],
            "distractor_type": strategy, "distractors": [choices[i] for i in order if i != 0],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="literary_benchmark")
    ap.add_argument("--gold_prep", default="../corpus/gold_ner_prep.json",
                    help="gold_ner_prep.json with the canonical train/test excerpt assignment")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    prep = json.loads(Path(args.gold_prep).read_text(encoding="utf-8"))
    excerpts = prep["excerpts"]
    books, report = [], {"offset_alignment": {}, "skipped": {}}
    for name, excerpt in sorted(excerpts.items()):
        text = excerpt["text"]
        events = []
        for idx, row in enumerate(excerpt.get("gold_movements", [])):
            s, e = row.get("start"), row.get("end")
            if s is None or e is None or not (0 <= s < e <= len(text)):
                continue
            # The prep file records canonical excerpt-coordinate offsets.
            if text[s:e] != row["text"]:
                report["skipped"].setdefault(name, []).append(f"movement {idx}: text/offset mismatch")
                continue
            events.append({"start": s, "end": e, "text": text[s:e], "idx": idx})
        if len(events) < 4:
            report["skipped"][name] = "fewer than four aligned movement spans"; continue
        report["offset_alignment"][name] = {"counts": {"gold_prep": len(events)}}
        books.append({"name": name, "text": text, "events": events,
                      "split": excerpt.get("split")})
    if len(books) < 2: raise SystemExit("Need at least two paired novels")
    assignment = {s: sorted(b["name"] for b in books if b["split"] == s)
                  for s in ("train", "test")}
    if not assignment["train"] or not assignment["test"]:
        raise SystemExit("gold_prep must assign at least one excerpt to train and test")
    bookmap = {b["name"]: b for b in books}
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for strategy in ("random", "hard"):
        for split, names in assignment.items():
            rows = [q for n in names for q in make_questions(bookmap[n], split, args.seed, strategy)]
            write_jsonl(out_dir / f"{strategy}_{split}.jsonl", rows)
            summary[f"{strategy}_{split}"] = {"questions": len(rows), "novels": names}
    report.update({"seed": args.seed, "split_by_novel": assignment, "counts": summary,
                   "source": args.gold_prep,
                   "note_split": "Uses the canonical split metadata in gold_ner_prep.json: movement spans from train excerpts train LitBench; test excerpts are the golden held-out set.",
                   "note": "Hard distractors prioritize sentences with action/spatial cues; sentences overlapping any annotated span are excluded."})
    (out_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
