# Literary movement benchmark

The source `gold_ner_prep.json` is not present in this checkout. The builder uses the available gold span lists in `corpus/annotations/consensus_final_*.json` and their paired `corpus/texts/*_excerpt.txt` files (`text`, `start_ind`, `end_ind`). It records exact, inclusive-end, whitespace, relocated, and failed alignment counts in `literary_benchmark/build_report.json`. Spans that cannot be aligned are omitted.

For each aligned movement span, the correct choice is the exact annotated text. Three unique distractors are sentence spans from the same novel, excluding every sentence that overlaps any annotated movement. Random distractors are sampled from this pool. Hard distractors prioritize simple action-verb and spatial-preposition cues, then sample among the highest-ranked candidates. The passage supplied to the model is a 2,400-character window around the gold span so it fits common model context limits. This means all choice sentences come from that novel, but distractors need not appear in the displayed local window.

Novels, not questions, are split 80/20 using seed 17. The same split is used for both strategies. The generated set currently has 26 train novels (955 questions) and 6 held-out novels (381 questions). The split assignment and offset diagnostics are in `literary_benchmark/build_report.json`.

## Commands

From `lora/`:

```bash
python3 build_literary_benchmark.py --seed 17 --out_dir literary_benchmark
python3 -m unittest test_literary_benchmark.py
python3 evaluate_literary.py --data literary_benchmark/random_test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out literary_benchmark/predictions_random_qwen.jsonl \
  --summary literary_benchmark/results_random_qwen.json
python3 evaluate_literary.py --data literary_benchmark/hard_test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out literary_benchmark/predictions_hard_qwen.jsonl \
  --summary literary_benchmark/results_hard_qwen.json
```

Use `--adapter PATH` to evaluate a PEFT adapter, `--few_shot N` to prepend N examples from the matching `*_train.jsonl`, `--seed` to control selection and random-chance predictions, and `--limit N` for a smoke run. `--model random` runs the uniform-choice baseline without loading ML libraries.

## Preliminary result

The deterministic uniform-choice baseline scored 25.98% on random distractors and 22.31% on hard distractors over 381 questions each; the expected chance rate is 25%. Sampling variation and answer-position balance account for small deviations. Per-question predictions and per-novel breakdowns are in the corresponding `results_*_chance.json` and `predictions_*_chance.jsonl` files. A Qwen run was not possible in the current environment because PyTorch is not installed; install `lora/requirements.txt` dependencies and ensure the model is available before running the commands above.

## Limitations

These are available consensus annotations, not `gold_ner_prep.json`; annotation quality and excerpt-version mismatches remain possible. Offset relocation is nearest exact-text search within 300 characters. The hard heuristic is intentionally simple and is not a human difficulty annotation. The random baseline is one seeded draw, not an analytically exact 25% in a finite sample. No fine-tuned StepGame checkpoint for Qwen was available in the inspected repository; the existing adapters are Pythia checkpoints, so comparing one would confound training data with base model.
