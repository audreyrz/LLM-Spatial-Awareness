# Qwen LoRA fine-tuning on StepGame

`train_stepgame.py` downloads `ZhengyanShi/StepGame`, creates a deterministic validation slice from its official `train` split, and fine-tunes Qwen with LoRA. The official test split is not read during training. The prompt and answer format match `evaluate_spatial_benchmarks.py`.

Run from `lora/` with the dependencies in `requirements.txt` installed and Hugging Face access configured:

```bash
python3 train_stepgame.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out_dir runs/qwen-stepgame \
  --seed 17
```

The best validation-loss adapter is saved to `runs/qwen-stepgame/best`. For a quick pipeline check, pass `--limit_train 128 --limit_val 32 --epochs 1`; those caps are only for debugging, not a research result. Training supports `--batch_size`, `--grad_accum`, `--lr`, `--epochs`, and `--max_len`.

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
