#!/usr/bin/env python3
"""Evaluate Laya's typed-choice model on Task B (next-event ranking), v2.

Zero-shot: Laya is a pretrained decision model; nothing is trained here.

Changes vs. v1:
  * Clearer task statement (names the fields, says 3 distractors come from
    elsewhere in the book, says the answer must follow the END of the passage).
  * Option descriptions are written as conditions, not bare restatements, and
    candidates are no longer duplicated in the state (--prompt_style v2).
    --prompt_style orig reproduces the v1 encoding for comparison.
  * Candidate-order permutation: each example is scored under several cyclic
    shifts of the options and votes are aggregated in original-candidate space,
    which cancels first-option (position) bias.
  * Baselines: gold label distribution and always-A / majority accuracy.
  * Optional "planted" sanity condition (--sanity): the gold phrase is appended
    to the narrative as "Next movement: <phrase>". A model that can read the
    interface should score near 100% here. If it does not, the setup is broken.
  * Paired bootstrap of each condition vs. context.
  * --debug prints raw Laya outputs for the first few items.
  * --max_narrative_chars keeps the END of long narratives.

Install the optional dependency with ``python -m pip install laya``.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import CONDITIONS, LETTERS, read_jsonl, write_jsonl

PLANTED = "planted"


# ----------------------------------------------------------------- data utils
def make_donors(examples):
    """Choose a distant narrative from the same book, as in evaluate.py."""
    groups = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex.get("book"), []).append(i)
    donors = [None] * len(examples)
    for indices in groups.values():
        for pos, index in enumerate(indices):
            if len(indices) > 1:
                donor_index = indices[(pos + max(1, len(indices) // 2)) % len(indices)]
            else:
                donor_index = (index + max(1, len(examples) // 2)) % len(examples)
            donors[index] = examples[donor_index]
    return donors


def get_narrative(ex, donor, condition):
    if condition == "context":
        return ex["preceding"]
    if condition == "nocontext":
        return "(not provided)"
    if condition == PLANTED:
        gold = ex["candidates"][LETTERS.index(ex["label"])]
        return f"{ex['preceding']}\n\nNext movement: {gold}"
    return donor["preceding"]  # shuffled


def trim(text, max_chars):
    """Keep the END of the narrative: the next event follows the end."""
    if max_chars and len(text) > max_chars:
        return text[-max_chars:]
    return text


# --------------------------------------------------------------- prompt build
INSTR_V2 = (
    "The field `preceding_narrative` is a passage from a book. Options A-D are "
    "movement phrases. Exactly one option is the movement that is annotated as "
    "happening NEXT, immediately after the end of the passage; the other three "
    "are movements from other parts of the same book. Choose the option whose "
    "movement most plausibly follows directly from the end of the passage. "
    "If the passage is '(not provided)', pick your best guess."
)
INSTR_ORIG = (
    "Which candidate is the next annotated movement phrase in the "
    "book after the preceding narrative? Select the candidate that "
    "best matches the narrative and event sequence."
)


def build_input(narrative, phrases, style):
    """Return (state, question) for one example with phrases in presented order."""
    state = {"preceding_narrative": narrative}
    if style == "orig":
        state.update({f"candidate_{l}": p for l, p in zip(LETTERS, phrases)})
        instructions = INSTR_ORIG
        criteria = {l: f"The movement phrase is: {p}" for l, p in zip(LETTERS, phrases)}
    else:
        instructions = INSTR_V2
        criteria = {
            l: f"Choose {l} if the next movement after the passage is: {p}"
            for l, p in zip(LETTERS, phrases)
        }
    question = {"next_event": {"type": "choice", "instructions": instructions,
                               "criteria": criteria}}
    return state, question


def laya_choice(agent, state, question):
    result = agent.predict(state, question)
    answer = result["answers"]["next_event"]
    pred = answer.get("choice")
    if pred not in LETTERS:
        raise ValueError(f"Laya returned unexpected choice {pred!r}; expected one of {LETTERS}")
    return pred, answer.get("confidence"), result


# ------------------------------------------------------------------ statistics
def bootstrap_ci(correct, n_boot, seed):
    if not len(correct):
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_diff(a, b, n_boot, seed):
    """Paired bootstrap of mean(a) - mean(b) over the same examples."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


# ------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--sanity", action="store_true",
                        help="also run the planted-answer sanity condition")
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="Laya checkpoint or Hugging Face repo")
    parser.add_argument("--prompt_style", default="v2", choices=["v2", "orig"])
    parser.add_argument("--n_perms", type=int, default=4, choices=[1, 2, 3, 4],
                        help="candidate orderings per example (cyclic shifts); 1 = original order only")
    parser.add_argument("--max_narrative_chars", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N examples")
    parser.add_argument("--debug", type=int, default=0,
                        help="print raw Laya output for the first N calls")
    parser.add_argument("--out", default="laya_results.json")
    parser.add_argument("--pred_out", default="laya_predictions.jsonl")
    args = parser.parse_args()
    if "context" not in args.conditions:
        args.conditions.insert(0, "context")
    if args.sanity:
        args.conditions.append(PLANTED)

    try:
        import laya
    except ImportError as exc:
        raise SystemExit("Missing optional dependency; install with: python -m pip install laya") from exc

    examples = read_jsonl(Path(args.data_dir) / f"task_b_{args.split}.jsonl")
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise SystemExit(f"No Task B examples found for split {args.split!r} in {args.data_dir}")
    if any(len(ex.get("candidates", [])) != len(LETTERS) for ex in examples):
        raise SystemExit("Expected exactly four candidates per Task B example")

    n = len(examples)
    gold_idx = np.array([LETTERS.index(ex["label"]) for ex in examples])
    gold_dist = {l: int((gold_idx == i).sum()) for i, l in enumerate(LETTERS)}
    baselines = {
        "chance": 1 / len(LETTERS),
        "gold_label_distribution": gold_dist,
        "always_" + "_".join(LETTERS[:1]): float((gold_idx == 0).mean()),
        "majority_class": float(max(gold_dist.values()) / n),
    }
    print(f"gold label distribution: {gold_dist}  |  always-A acc={baselines['always_A']:.3f}")

    donors = make_donors(examples)
    agent = laya.load(args.model)
    rng = np.random.default_rng(args.seed)
    n_cand = len(LETTERS)
    results, prediction_rows, correct_by_cond = {}, [], {}
    debug_left = args.debug

    for condition in args.conditions:
        agg_correct = np.zeros(n)
        orig_order_correct = np.zeros(n)
        pos_counts = {l: 0 for l in LETTERS}        # letters as presented to Laya
        final_counts = {l: 0 for l in LETTERS}      # aggregated, original labels
        perm_correct, n_perm_calls = 0, 0

        for i, ex in enumerate(examples):
            narrative = trim(get_narrative(ex, donors[i], condition), args.max_narrative_chars)
            votes = np.zeros(n_cand)
            conf_sum = np.zeros(n_cand)
            for shift in range(args.n_perms):
                order = [(p + shift) % n_cand for p in range(n_cand)]  # position -> original idx
                phrases = [ex["candidates"][j] for j in order]
                state, question = build_input(narrative, phrases, args.prompt_style)
                pred, conf, raw = laya_choice(agent, state, question)
                if debug_left > 0:
                    print(f"[debug] cond={condition} id={ex['id']} shift={shift} -> {raw}")
                    debug_left -= 1
                orig_j = order[LETTERS.index(pred)]
                votes[orig_j] += 1
                conf_sum[orig_j] += conf if isinstance(conf, (int, float)) else 0.0
                pos_counts[pred] += 1
                n_perm_calls += 1
                perm_correct += int(orig_j == gold_idx[i])
                if shift == 0:
                    orig_order_correct[i] = float(orig_j == gold_idx[i])
                prediction_rows.append({
                    "model": "laya", "checkpoint": args.model, "condition": condition,
                    "task": "B", "id": ex["id"], "shift": shift,
                    "pred_position": pred, "pred_original": LETTERS[orig_j],
                    "gold": ex["label"], "confidence": conf,
                })
            # aggregate: votes, then summed confidence, then random tie-break
            tiebreak = rng.random(n_cand)
            final = max(range(n_cand), key=lambda j: (votes[j], conf_sum[j], tiebreak[j]))
            final_counts[LETTERS[final]] += 1
            agg_correct[i] = float(final == gold_idx[i])

        correct_by_cond[condition] = agg_correct
        results[condition] = {
            "n": n,
            "accuracy": float(agg_correct.mean()),
            "ci95": bootstrap_ci(agg_correct, args.n_boot, args.seed),
            "accuracy_original_order_only": float(orig_order_correct.mean()),
            "accuracy_per_permuted_call": perm_correct / n_perm_calls,
            "chance": 1 / n_cand,
            "pred_position_distribution": pos_counts,
            "pred_final_label_distribution": final_counts,
        }
        print(f"[{condition:9s}] n={n} acc(agg, {args.n_perms} orders)={agg_correct.mean():.3f} "
              f"CI {results[condition]['ci95']} | orig-order acc={orig_order_correct.mean():.3f} "
              f"| position dist {pos_counts}")

    comparisons = {}
    ctx = correct_by_cond["context"]
    for cond, arr in correct_by_cond.items():
        if cond != "context":
            comparisons[f"context_minus_{cond}"] = paired_diff(ctx, arr, args.n_boot, args.seed)
            print(f"paired context - {cond}: {comparisons[f'context_minus_{cond}']}")

    output = {
        "meta": {
            "model": args.model,
            "split": args.split,
            "n_examples": n,
            "note": ("Task B is next annotated event, not necessarily next physical movement. "
                     "Zero-shot Laya typed-choice; candidates scored under cyclic order shifts "
                     "and aggregated by vote to remove position bias."),
            "args": vars(args),
        },
        "baselines": baselines,
        "results": results,
        "paired_comparisons": comparisons,
    }
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_jsonl(args.pred_out, prediction_rows)
    print(f"wrote {args.out} and {args.pred_out}")


if __name__ == "__main__":
    main()