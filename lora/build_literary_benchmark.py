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

from build_data import align, load_annotations
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
    ap.add_argument("--text_dir", default="../corpus/texts")
    ap.add_argument("--ann_dir", default="../corpus/annotations")
    ap.add_argument("--out_dir", default="literary_benchmark")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test_fraction", type=float, default=0.2)
    args = ap.parse_args()
    text_dir, ann_dir, out_dir = Path(args.text_dir), Path(args.ann_dir), Path(args.out_dir)
    txts = {p.stem[:-8]: p for p in text_dir.glob("*_excerpt.txt")}
    anns = {p.stem.removeprefix("consensus_final_"): p for p in ann_dir.glob("consensus_final_*.json")}
    mapping = {"murderOnTheYukon": "murderOnTheYukonQuest", "tarzan": "tarzanTheTerrible"}
    books, report = [], {"offset_alignment": {}, "skipped": {}}
    pairs = [(n, n) for n in anns.keys() & txts.keys()]
    pairs += [(a, t) for a, t in mapping.items() if a in anns and t in txts]
    for ann_name, text_name in sorted(pairs):
        text = txts[text_name].read_text(encoding="utf-8")
        raw, bad = load_annotations(anns[ann_name], "text", "start_ind", "end_ind")
        events, statuses = [], {}
        for row in raw:
            s, e, status = align(text, row)
            statuses[status] = statuses.get(status, 0) + 1
            if s is not None: events.append({"start": s, "end": e, "text": text[s:e], "idx": row["src_idx"]})
        events.sort(key=lambda x: (x["start"], x["end"]))
        clean, last_end = [], -1
        for e in events:
            if e["start"] >= last_end:
                clean.append(e); last_end = e["end"]
        if len(clean) < 4:
            report["skipped"][ann_name] = "fewer than four aligned nonoverlapping spans"; continue
        report["offset_alignment"][ann_name] = {"counts": statuses, "unparseable": bad, "dropped_overlaps": len(events)-len(clean)}
        books.append({"name": ann_name, "text": text, "events": clean})
    if len(books) < 2: raise SystemExit("Need at least two paired novels")
    rng = random.Random(args.seed); order = sorted(b["name"] for b in books); rng.shuffle(order)
    ntest = max(1, min(len(order)-1, round(len(order)*args.test_fraction)))
    test_names = set(order[:ntest]); assignment = {"train": sorted(set(order)-test_names), "test": sorted(test_names)}
    bookmap = {b["name"]: b for b in books}
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for strategy in ("random", "hard"):
        for split, names in assignment.items():
            rows = [q for n in names for q in make_questions(bookmap[n], split, args.seed, strategy)]
            write_jsonl(out_dir / f"{strategy}_{split}.jsonl", rows)
            summary[f"{strategy}_{split}"] = {"questions": len(rows), "novels": names}
    report.update({"seed": args.seed, "split_by_novel": assignment, "counts": summary,
                   "source": "corpus/annotations/consensus_final_*.json and corpus/texts/*_excerpt.txt",
                   "note": "Hard distractors prioritize sentences with action/spatial cues; sentences overlapping any annotated span are excluded."})
    (out_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
