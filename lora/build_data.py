#!/usr/bin/env python3
"""Validate offsets and build Task A / Task B examples from a COLLECTION of texts.

Corpus layout: one <name>.txt and one <name>.json per text (matched by file stem).
Each JSON is a list of {"text", "start_ind", "end_ind"}.

The train/val/test split is made BY TEXT: every example from a given text lands
in the same split, so the model is always evaluated on books it never saw.

Outputs (in --out_dir)
  task_a_{train,val,test}.jsonl   masked movement-phrase reconstruction
  task_b_{train,val,test}.jsonl   next-annotated-event ranking (4 candidates)
  build_report.json               per-text validation, split assignment, counts,
                                  trivial-baseline diagnostics for Task B

Design choices that matter for validity
  * Split unit = text. Context windows never cross a text boundary.
  * Task B distractors default to phrases annotated elsewhere in the SAME text
    (so character names / setting cannot give the answer away), length-matched,
    and matched on "does the phrase literally appear in the preceding window".
    Falls back to the same split's pool for texts with too few events.
  * Task B is "next ANNOTATED event", not necessarily next physical movement.
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from common import LETTERS, normalize_phrase, write_jsonl

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
# loading + offset validation
# ----------------------------------------------------------------------------
def load_annotations(path, text_key, start_key, end_key):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists:
            return [], 1
        data = max(lists, key=len)
    rows, bad = [], 0
    for i, r in enumerate(data):
        try:
            rows.append(
                {
                    "src_idx": i,
                    "text": str(r[text_key]),
                    "start": int(r[start_key]),
                    "end": int(r[end_key]),
                }
            )
        except (KeyError, TypeError, ValueError):
            bad += 1
    return rows, bad


def align(text, a, reloc_window=300):
    """Return (start, end, status). Source text is authoritative."""
    s, e, t = a["start"], a["end"], a["text"]
    n = len(text)
    if 0 <= s < e <= n and text[s:e] == t:
        return s, e, "exact"
    if 0 <= s and e + 1 <= n and s < e + 1 and text[s : e + 1] == t:
        return s, e + 1, "inclusive_end"
    if 0 <= s < e <= n and " ".join(text[s:e].split()) == " ".join(t.split()):
        return s, e, "whitespace_only"
    if not t.strip():
        return None, None, "failed"
    lo, hi = max(0, s - reloc_window), min(n, e + reloc_window)
    best = None
    for m in re.finditer(re.escape(t), text[lo:hi]):
        pos = lo + m.start()
        if best is None or abs(pos - s) < abs(best - s):
            best = pos
    if best is not None:
        return best, best + len(t), "relocated"
    return None, None, "failed"


def prepare_text(name, txt_path, json_path, args):
    """Validate one text. Returns dict(name, text, events, report) or None."""
    # newline="" preserves \r\n so offsets are not silently shifted
    with open(txt_path, encoding=args.encoding, newline="") as f:
        text = f.read()
    raw, bad_rows = load_annotations(json_path, args.text_key, args.start_key, args.end_key)

    status, shifts, events = Counter(), [], []
    for a in raw:
        s, e, st = align(text, a)
        status[st] += 1
        if st == "failed":
            continue
        if st == "relocated":
            shifts.append(s - a["start"])
        events.append({"start": s, "end": e, "text": text[s:e]})

    n_long = sum(len(e["text"]) > args.max_span_chars for e in events)
    events = [e for e in events if len(e["text"]) <= args.max_span_chars and e["text"].strip()]
    events.sort(key=lambda e: (e["start"], e["end"]))
    clean, n_overlap, last_end = [], 0, -1
    for e in events:
        if e["start"] < last_end:
            n_overlap += 1
            continue
        clean.append(e)
        last_end = e["end"]
    for i, e in enumerate(clean):
        e["idx"] = i
        e["book"] = name

    report = {
        "n_chars": len(text),
        "n_annotations_raw": len(raw),
        "unparseable_rows": bad_rows,
        "alignment": dict(status),
        "dropped_too_long": n_long,
        "dropped_overlap": n_overlap,
        "n_events_used": len(clean),
    }
    if shifts and len(shifts) > 0.2 * max(1, len(raw)):
        report["warning"] = (
            f"{len(shifts)} spans needed relocation (median shift {statistics.median(shifts)}): "
            "offsets may be systematically off (\\r\\n vs \\n, UTF-16/byte offsets, different text version)"
        )
        print(f"WARNING [{name}]: {report['warning']}")
    if args.strict and status["exact"] != len(raw):
        raise SystemExit(f"--strict: [{name}] offsets not all exact: {dict(status)}")
    return {"name": name, "text": text, "events": clean, "report": report}


# ----------------------------------------------------------------------------
# split by text
# ----------------------------------------------------------------------------
def assign_splits(books, fracs, seed, split_file):
    """Return {split: [book names]}. Whole texts only."""
    names = [b["name"] for b in books]
    if split_file:
        spec = json.loads(Path(split_file).read_text())
        listed = [n for s in SPLITS for n in spec.get(s, [])]
        unknown = sorted(set(listed) - set(names))
        missing = sorted(set(names) - set(listed))
        if unknown:
            raise SystemExit(f"split_file lists unknown texts: {unknown}")
        if missing:
            print(f"WARNING: texts not in split_file are excluded: {missing}")
        if len(listed) != len(set(listed)):
            raise SystemExit("split_file lists a text more than once")
        return {s: list(spec.get(s, [])) for s in SPLITS}

    if len(books) < 3:
        raise SystemExit(f"need >= 3 texts to split by text, found {len(books)}")
    rng = random.Random(seed)
    order = sorted(books, key=lambda b: b["name"])
    rng.shuffle(order)
    total = sum(len(b["events"]) for b in books)
    fsum = sum(fracs)
    target = {s: f / fsum * total for s, f in zip(SPLITS, fracs)}
    count = {s: 0 for s in SPLITS}
    out = {s: [] for s in SPLITS}
    # greedy: give each text to the split furthest below its event-count target
    # (an empty split always has the largest deficit, so none stays empty)
    for b in order:
        s = max(SPLITS, key=lambda s: (target[s] - count[s]) / target[s])
        out[s].append(b["name"])
        count[s] += len(b["events"])
    return out


# ----------------------------------------------------------------------------
# context windows (clipped to the text, cut at whitespace)
# ----------------------------------------------------------------------------
_WS = re.compile(r"\s")


def clip_before(text, pos, n):
    s = max(0, pos - n)
    if s > 0:
        m = _WS.search(text, s, pos)
        if m:
            s = m.end()
    return text[s:pos]


def clip_after(text, pos, n):
    e = min(len(text), pos + n)
    if e < len(text):
        idx = max(text.rfind(" ", pos, e), text.rfind("\n", pos, e))
        if idx > pos:
            e = idx
    return text[pos:e]


# ----------------------------------------------------------------------------
# Task B distractors
# ----------------------------------------------------------------------------
def pick_distractors(rng, target, pools, preceding, k=3, topk=20):
    """pools: list of candidate lists in priority order (same text, then same split).
    Return (list of k distractor strings | None, used_fallback)."""
    tnorm = normalize_phrase(target["text"])
    pl = preceding.lower()

    def appears(ev):
        return ev["text"].lower() in pl

    t_in = appears(target)
    for level, pool in enumerate(pools):
        seen, cands = {tnorm}, []
        for ev in pool:
            nn = normalize_phrase(ev["text"])
            if ev is target or nn in seen or not nn:
                continue
            seen.add(nn)
            cands.append(ev)
        if len(cands) < k:
            continue
        matched = [c for c in cands if appears(c) == t_in]
        fallback = len(matched) < k
        if fallback:
            matched = cands
        tlen = len(target["text"])
        matched.sort(key=lambda c: abs(len(c["text"]) - tlen))
        near = matched[: max(topk, k)]
        return [c["text"] for c in rng.sample(near, k)], (fallback or level > 0)
    return None, False


def heuristic_baselines(b_rows):
    """Expected accuracy of shortcut heuristics (for sanity-checking Task B)."""
    if not b_rows:
        return {}
    longest, in_win = [], []
    for r in b_rows:
        lens = [len(c) for c in r["candidates"]]
        ti = LETTERS.index(r["label"])
        mx = max(lens)
        longest.append((1 / lens.count(mx)) if lens[ti] == mx else 0.0)
        pl = r["preceding"].lower()
        hits = [i for i, c in enumerate(r["candidates"]) if c.lower() in pl]
        in_win.append((1 / len(hits)) if (hits and ti in hits) else (0.25 if not hits else 0.0))
    return {
        "chance": 0.25,
        "longest_candidate_acc": round(sum(longest) / len(longest), 4),
        "appears_in_window_acc": round(sum(in_win) / len(in_win), 4),
        "label_distribution": dict(Counter(r["label"] for r in b_rows)),
    }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text_dir", required=True, help="folder with <name>.txt files")
    ap.add_argument("--ann_dir", default=None, help="folder with <name>.json files (default: same as --text_dir)")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--start_key", default="start_ind")
    ap.add_argument("--end_key", default="end_ind")
    ap.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15], help="target fractions of EVENTS, assigned by whole text")
    ap.add_argument("--split_file", default=None, help='optional JSON {"train":[names],"val":[...],"test":[...]} to fix the assignment')
    ap.add_argument("--min_events_per_text", type=int, default=8)
    ap.add_argument("--a_window", type=int, default=600, help="chars on each side (Task A)")
    ap.add_argument("--b_window", type=int, default=1200, help="chars of preceding narrative (Task B)")
    ap.add_argument("--min_context_chars", type=int, default=150, help="skip Task B examples with less preceding text")
    ap.add_argument("--max_span_chars", type=int, default=200)
    ap.add_argument("--b_draws_train", type=int, default=3, help="distractor resamples per train event (val/test use 1)")
    ap.add_argument("--distractors", choices=["text", "split"], default="text",
                    help="'text': prefer distractors from the same text; 'split': pool over all texts in the split")
    ap.add_argument("--strict", action="store_true", help="abort unless every offset is exact")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    ann_dir = Path(args.ann_dir) if args.ann_dir else text_dir
    txts = {p.stem: p for p in sorted(text_dir.glob("*.txt"))}
    jsons = {p.stem: p for p in sorted(ann_dir.glob("*.json"))}
    paired = sorted(set(txts) & set(jsons))
    unmatched = {"txt_without_json": sorted(set(txts) - set(jsons)), "json_without_txt": sorted(set(jsons) - set(txts))}
    if not paired:
        raise SystemExit(f"no <name>.txt / <name>.json pairs found ({unmatched})")
    if any(unmatched.values()):
        print("WARNING: unmatched files ignored:", unmatched)

    # ---- 1. validate each text -------------------------------------------
    books, text_reports, dropped_small = [], {}, []
    for name in paired:
        b = prepare_text(name, txts[name], jsons[name], args)
        text_reports[name] = b["report"]
        if len(b["events"]) < args.min_events_per_text:
            dropped_small.append(name)
            continue
        books.append(b)
    if dropped_small:
        print(f"WARNING: dropped texts with < {args.min_events_per_text} usable events: {dropped_small}")

    # ---- 2. split by text ------------------------------------------------
    assignment = assign_splits(books, args.split, args.seed, args.split_file)
    by_name = {b["name"]: b for b in books}

    # ---- 3. build tasks --------------------------------------------------
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    report = {
        "unmatched_files": unmatched,
        "texts_dropped_too_few_events": dropped_small,
        "texts": text_reports,
        "split_assignment": assignment,
        "splits": {},
        "task_b_baselines": {},
        "task_b_distractor_fallbacks": 0,
        "args": vars(args),
    }
    for split in SPLITS:
        split_events = [e for n in assignment[split] for e in by_name[n]["events"]]
        a_rows, b_rows, skipped_b = [], [], 0
        draws = args.b_draws_train if split == "train" else 1
        for name in assignment[split]:
            b = by_name[name]
            text, evs = b["text"], b["events"]
            for ev in evs:
                a_rows.append(
                    {
                        "task": "A",
                        "id": f"A-{name}-{ev['idx']}",
                        "split": split,
                        "book": name,
                        "event_idx": ev["idx"],
                        "before": clip_before(text, ev["start"], args.a_window),
                        "after": clip_after(text, ev["end"], args.a_window),
                        "target": ev["text"],
                    }
                )
                prec = clip_before(text, ev["start"], args.b_window)
                if len(prec.strip()) < args.min_context_chars:
                    skipped_b += 1
                    continue
                pools = [evs, split_events] if args.distractors == "text" else [split_events]
                for d in range(draws):
                    dis, fb = pick_distractors(rng, ev, pools, prec)
                    if dis is None:
                        skipped_b += 1
                        break
                    report["task_b_distractor_fallbacks"] += int(fb)
                    cands = dis + [ev["text"]]
                    rng.shuffle(cands)
                    b_rows.append(
                        {
                            "task": "B",
                            "id": f"B-{name}-{ev['idx']}-{d}",
                            "split": split,
                            "book": name,
                            "event_idx": ev["idx"],
                            "preceding": prec,
                            "candidates": cands,
                            "label": LETTERS[cands.index(ev["text"])],
                            "target": ev["text"],
                        }
                    )
        write_jsonl(out_dir / f"task_a_{split}.jsonl", a_rows)
        write_jsonl(out_dir / f"task_b_{split}.jsonl", b_rows)
        report["splits"][split] = {
            "texts": assignment[split],
            "events": len(split_events),
            "task_a_examples": len(a_rows),
            "task_b_examples": len(b_rows),
            "task_b_events_skipped": skipped_b,
            "mean_span_chars": round(statistics.mean(len(e["text"]) for e in split_events), 1) if split_events else None,
        }
        report["task_b_baselines"][split] = heuristic_baselines(b_rows)

    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"split_assignment": assignment, "splits": report["splits"]}, indent=2))
    print("Task B shortcut baselines:", json.dumps(report["task_b_baselines"], indent=2))


if __name__ == "__main__":
    main()
