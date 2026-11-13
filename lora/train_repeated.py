#!/usr/bin/env python3
"""Run the same LoRA training recipe several times with independent seeds.

Example: python train_repeated.py --repeats 5 --seed 100 --out_root runs/repeats
Training arguments after the wrapper options are forwarded to train_lora.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out_root", default="runs/repeated")
    args, forwarded = parser.parse_known_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    root = Path(args.out_root)
    trainer = Path(__file__).with_name("train_lora.py")
    for run in range(args.repeats):
        seed = args.seed + run
        out_dir = root / f"run_{run + 1:02d}_seed_{seed}"
        command = [sys.executable, str(trainer), *forwarded,
                   "--seed", str(seed), "--out_dir", str(out_dir)]
        print(f"\n=== run {run + 1}/{args.repeats}: seed={seed}, out={out_dir} ===", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
