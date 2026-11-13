# Literary movement benchmark

LitBench reads movement spans and the canonical split from `corpus/gold_ner_prep.json`. The 26 excerpts marked `train` supply training questions; the six excerpts marked `test` are the golden held-out test set. The same split applies to random and hard distractor variants. The benchmark builder verifies each stored movement span against the excerpt text before using it.

Each question has exactly four choices: the annotated movement span and three unique sentence distractors from the same excerpt. Hard distractors prioritize sentences with simple action and spatial cues. The passage is a 2,400-character window around the gold span. Choice order is shuffled per example and evaluation seed; predictions are scored against the remapped answer letter.

The main Task A/Task B movement dataset can use the same golden test assignment while carving validation books from the 26 training excerpts:

```bash
python3 build_data.py --text_dir ../corpus/texts --ann_dir ../corpus/annotations \
  --out_dir data --gold_prep ../corpus/gold_ner_prep.json --seed 42
python3 build_literary_benchmark.py --gold_prep ../corpus/gold_ner_prep.json \
  --out_dir literary_benchmark --seed 42
```

Repeat LoRA training with independent seeds and separate adapter directories:

```bash
python3 train_repeated.py --repeats 5 --seed 100 --out_root runs/repeated \
  --data_dir data --model Qwen/Qwen2.5-0.5B-Instruct
```

Evaluate LitBench on the golden set with `evaluate_literary.py --data literary_benchmark/random_test.jsonl`. Choose `hard_test.jsonl` for hard distractors. Pass `--adapter PATH` to score a trained adapter; omit it for the base model. `--seed` controls per-example option order.

`probe_spatial.py` extracts the final-layer contextual representations of the ordered entity pair in each StepGame question, then trains a linear classifier to predict the spatial relation. It reports base-model and optional adapter probe accuracy on the official test split. This measures relation information in entity-pair representations; it is a diagnostic, separate from generative benchmark accuracy.
