#!/usr/bin/env python
"""GAIA under a stated boundary: text-only questions, closed book, exact-match final answer.

GAIA's own protocol allows browsing, tools and file attachments; this harness has none of them,
so what is scored is the strict lower bound a model reaches from the question text alone. Rows
with a file attachment are excluded (the model cannot read what it is not given), and the match
follows GAIA's normalisation: numbers compare numerically, lists element-wise, strings after
lowercasing and article/punctuation stripping. None of this is a GAIA leaderboard score and it
must not be quoted as one.

  python scripts/gaia_eval.py --model hf:data/baselines/Qwen3-0.6B \
      --data data/public/gaia-validation.jsonl --out runs/gaia/qwen3-06b.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path


def normalise_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalise_string(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    return " ".join(word for word in text.split() if word not in ("the", "a", "an"))


def answers_match(predicted: str, gold: str) -> bool:
    """GAIA's comparison: numeric when the gold is numeric, element-wise for comma lists."""
    gold = gold.strip()
    gold_number = normalise_number(gold)
    if gold_number is not None:
        predicted_number = normalise_number(predicted)
        return predicted_number is not None and abs(predicted_number - gold_number) < 1e-6
    if "," in gold:
        gold_parts = [part.strip() for part in gold.split(",")]
        predicted_parts = [part.strip() for part in predicted.split(",")]
        return (len(gold_parts) == len(predicted_parts)
                and all(answers_match(p, g) for p, g in zip(predicted_parts, gold_parts)))
    return normalise_string(predicted) == normalise_string(gold)


PROMPT = ("Answer the question. Reply with ONLY the final answer — a number, a few words, or a "
          "comma-separated list — and nothing else.\n\nQuestion: {question}\nFinal answer:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="hf:<dir> or lora:<dir>|<adapter>")
    ap.add_argument("--data", default="data/public/gaia-validation.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--levels", default="1,2,3")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    levels = {int(level) for level in args.levels.split(",")}
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    rows = [row for row in rows
            if not row.get("file_name") and int(row.get("Level", 0)) in levels]

    kind, location = args.model.split(":", 1)
    base = location.split("|")[0]
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    if kind == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, location.split("|")[1]).eval()
    model.config.use_cache = True

    per_level: dict[int, list[bool]] = {}
    records = []
    started = time.time()
    for row in rows:
        question = row["Question"]
        messages = [{"role": "user", "content": PROMPT.format(question=question)}]
        ids = None
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                rendered = tok.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=False, **kwargs)
                ids = tok(rendered, return_tensors="pt").input_ids.to(args.device)
                break
            except (ValueError, TypeError, AttributeError):
                continue
        if ids is None:
            ids = tok(PROMPT.format(question=question),
                      return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        predicted = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        predicted = predicted.splitlines()[0].strip() if predicted else ""
        correct = answers_match(predicted, row["Final answer"])
        per_level.setdefault(int(row["Level"]), []).append(correct)
        records.append({"task_id": row.get("task_id"), "level": row["Level"],
                        "predicted": predicted, "gold": row["Final answer"],
                        "correct": correct})

    summary = {
        "model": args.model, "rows": len(rows),
        "claim_boundary": "text-only questions (file-attachment rows excluded), closed book, no "
                          "browsing or tools, GAIA-normalised exact match on the final answer; a "
                          "strict lower bound, not a GAIA leaderboard score",
        "accuracy": round(100 * sum(r["correct"] for r in records) / max(len(records), 1), 1),
        "per_level": {level: round(100 * sum(flags) / len(flags), 1)
                      for level, flags in sorted(per_level.items())},
        "rows_per_level": {level: len(flags) for level, flags in sorted(per_level.items())},
        "seconds": round(time.time() - started, 1),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    print("GAIA_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
