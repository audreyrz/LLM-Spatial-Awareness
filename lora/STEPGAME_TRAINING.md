# Qwen LoRA fine-tuning on StepGame

`train_stepgame.py` downloads `ZhengyanShi/StepGame`, creates a deterministic validation slice from its official `train` split, and fine-tunes Qwen with LoRA. The official test split is not read during training. The prompt and answer format match `evaluate_spatial_benchmarks.py`.

Run from `lora/` with the dependencies in `requirements.txt` installed and Hugging Face access configured:

```bash
python3 train_stepgame.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out_dir runs/qwen-stepgame \
  --seed 17
```

The best validation-loss adapter is saved to `runs/qwen-stepgame/best`. For a quick pipeline check, pass `--limit_train 128 --limit_val 32 --epochs 1`; those caps are only for debugging, not a research result. Training supports `--batch_size`, `--grad_accum`, `--lr`, `--epochs`, `--max_len`, and `--max_steps`.

## Speed options

The trainer uses SDPA attention, disables the inference KV cache, and groups similarly sized examples into batches to reduce padding. For a faster first full-data run on a GPU, try:

```bash
python3 train_stepgame.py --epochs 1 --max_len 1024 --batch_size 8 --grad_accum 1
```

If that runs out of memory, reduce `--batch_size` to 4 or 2 and increase `--grad_accum` to keep a similar effective batch size. Shorter `--max_len` can truncate long contexts and affect results. For a quick runtime estimate, use `--max_steps 100`; that adapter is only a speed check and should not be used for the final transfer comparison. On CPU, Qwen fine-tuning will be slow; a CUDA GPU is strongly preferred.

## Evaluate the spatial training task

This compares the same base Qwen checkpoint and the new adapter on official StepGame test data:

```bash
python3 evaluate_spatial_benchmarks.py \
  --benchmark stepgame \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/qwen-stepgame/best \
  --out runs/qwen-stepgame/stepgame_test_results.json \
  --pred_out runs/qwen-stepgame/stepgame_test_predictions.jsonl
```

The evaluator samples exactly four relation options per item, retains the gold relation, and shuffles their order using `--seed`. The resulting score is conditional on that sampled four-choice set.

To compare how much spatial-relation information is linearly accessible from ordered entity-pair states before and after adaptation, run:

```bash
python3 probe_spatial.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/qwen-stepgame/best \
  --out runs/qwen-stepgame/entity_pair_probe.json
```

## Evaluate transfer to literary movement

Use the exact same held-out novels for the base and StepGame-adapted model:

```bash
python3 evaluate_literary.py \
  --data literary_benchmark/random_test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/qwen-stepgame/best \
  --out runs/qwen-stepgame/literary_random_predictions.jsonl \
  --summary runs/qwen-stepgame/literary_random_results.json

python3 evaluate_literary.py \
  --data literary_benchmark/hard_test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/qwen-stepgame/best \
  --out runs/qwen-stepgame/literary_hard_predictions.jsonl \
  --summary runs/qwen-stepgame/literary_hard_results.json
```

`evaluate_literary.py` reports the adapter's result. Run it once more per data file without `--adapter` for the base model comparison. Keep the benchmark seed, prompt, max context window, and model revision unchanged between runs. StepGame training and literary evaluation currently use distinct tasks; no literary test data enters training or checkpoint selection.
