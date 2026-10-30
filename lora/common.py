"""Shared helpers for the narrative-movement pilot (prompts, normalization, IO).

Prompt rendering lives here so that training and evaluation use *exactly* the
same templates.
"""
import json
import re
from collections import Counter
from pathlib import Path

MASK = "[MOVEMENT_SPAN]"
LETTERS = "ABCD"
CONDITIONS = ("context", "nocontext", "shuffled")

INSTR_A = (
    "Below is a passage from a novel. One movement-related phrase has been "
    f"replaced with {MASK}. Reconstruct the missing phrase exactly as it would "
    "appear in the text. Reply with the phrase only."
)
INSTR_B = (
    "Below is the narrative that precedes a point in a novel, followed by four "
    "candidate movement-related phrases. Exactly one candidate is the NEXT "
    "annotated movement phrase in the book after this narrative; the other "
    "three are annotated elsewhere in the book. Choose the correct one. "
    "Reply with a single letter (A, B, C, or D)."
)


# ----------------------------------------------------------------------------
# text normalization / metrics
# ----------------------------------------------------------------------------
def normalize_phrase(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_phrase(pred).split(), normalize_phrase(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
def user_prompt(ex: dict, condition: str = "context", donor: dict = None) -> str:
    """Render the user message.

    condition:
      context   - the example's own narrative
      nocontext - narrative removed (mask token only / "not provided")
      shuffled  - narrative taken from `donor`, a different example in the same
                  split (same length/style, wrong place in the book)
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)
    src = ex if condition == "context" else donor
    if condition == "shuffled" and donor is None:
        raise ValueError("shuffled condition needs a donor example")

    if ex["task"] == "A":
        passage = MASK if condition == "nocontext" else src["before"] + MASK + src["after"]
        return f"{INSTR_A}\n\nPassage:\n{passage}"

    narrative = "(not provided)" if condition == "nocontext" else src["preceding"]
    cands = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, ex["candidates"]))
    return (
        f"{INSTR_B}\n\nPreceding narrative:\n{narrative}\n\n"
        f"Candidates:\n{cands}\n\nAnswer:"
    )


def build_prompt_text(tokenizer, ex, condition="context", donor=None) -> str:
    content = user_prompt(ex, condition, donor)
    # Instruct models (e.g. Qwen) use their native chat format. Base models
    # such as Pythia have no chat template, so train and evaluate on the same
    # plain-text continuation prompt instead.
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return content.rstrip() + "\n"


def answer_text(ex: dict) -> str:
    return ex["target"] if ex["task"] == "A" else ex["label"]


# ----------------------------------------------------------------------------
# device
# ----------------------------------------------------------------------------
def pick_device_dtype():
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return "cuda", dtype
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32
