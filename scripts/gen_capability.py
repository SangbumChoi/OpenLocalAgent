#!/usr/bin/env python
"""Score general language ability before and after the agent fine-tuning recipe.

Specialising a model on one union corpus of tool calls is only free if it leaves the rest of the
model alone, and that is an empirical question rather than an assumption. This scores the same
checkpoint with and without the LoRA adapter on six standard multiple-choice benchmarks, using
length-normalised continuation likelihood — no generation, no parsing, so nothing here depends on
the chat template or the answer format the agent recipe taught.

  python scripts/gen_capability.py --model data/baselines/SmolLM2-135M-Instruct \
      --adapter runs/lora/smollm2-135m --tag smollm2-135m --rows 400
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

ROOT = Path("data/general")

# Each loader returns (context, [continuations], gold_index). The continuation is scored as a
# suffix of the context, which is the standard harness convention for these suites.
LETTERS = "ABCDE"


def rows_arc(record) -> tuple[str, list[str], int] | None:
    choices = record["choices"]
    texts, labels = list(choices["text"]), list(choices["label"])
    key = record["answerKey"]
    if key not in labels:
        return None
    return f"Question: {record['question']}\nAnswer:", [f" {t}" for t in texts], labels.index(key)


def rows_hellaswag(record) -> tuple[str, list[str], int] | None:
    endings = list(record["endings"])
    label = record.get("label")
    if label in (None, ""):
        return None
    context = f"{record['activity_label']}: {record['ctx']}"
    return context, [f" {e}" for e in endings], int(label)


def rows_openbookqa(record) -> tuple[str, list[str], int] | None:
    choices = record["choices"]
    labels = list(choices["label"])
    key = record["answerKey"]
    if key not in labels:
        return None
    return record["question_stem"], [f" {t}" for t in choices["text"]], labels.index(key)


def rows_winogrande(record) -> tuple[str, list[str], int] | None:
    """Winogrande scores the *shared* suffix under each substitution, so the context varies."""
    sentence, answer = record["sentence"], record.get("answer")
    if answer not in ("1", "2") or "_" not in sentence:
        return None
    prefix, suffix = sentence.split("_", 1)
    options = [record["option1"], record["option2"]]
    # Encoded as (context, continuation) pairs sharing one continuation string.
    return [prefix + option for option in options], [suffix, suffix], int(answer) - 1


def rows_mmlu(record) -> tuple[str, list[str], int] | None:
    choices = list(record["choices"])
    answer = int(record["answer"])
    lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{record['question']}\n{lines}\nAnswer:", [f" {LETTERS[i]}" for i in
                                                       range(len(choices))], answer


LOADERS = {"arc-easy": rows_arc, "arc-challenge": rows_arc, "hellaswag": rows_hellaswag,
           "openbookqa": rows_openbookqa, "winogrande": rows_winogrande, "mmlu": rows_mmlu}


def read_suite(name: str, limit: int) -> list[tuple]:
    import pyarrow.parquet as pq

    files = sorted((ROOT / name).rglob("*.parquet"))
    if not files:
        return []
    loader = LOADERS[name]
    out = []
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for record in batch.to_pylist():
                item = loader(record)
                if item is not None:
                    out.append(item)
                if limit and len(out) >= limit:
                    return out
    return out


@torch.no_grad()
def continuation_logprob(model, tok, context: str, continuation: str, device) -> tuple[float, int]:
    """Total and per-token log-probability of `continuation` following `context`."""
    context_ids = tok(context, return_tensors="pt").input_ids
    full_ids = tok(context + continuation, return_tensors="pt").input_ids
    if full_ids.shape[1] <= context_ids.shape[1]:
        return -math.inf, 1
    full_ids = full_ids[:, -1024:].to(device)
    start = max(1, context_ids.shape[1] - max(0, full_ids.shape[1] - 1024))
    logits = model(input_ids=full_ids).logits.float()
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = full_ids[0, 1:]
    span = logprobs[start - 1:, :].gather(1, targets[start - 1:].unsqueeze(1)).squeeze(1)
    return float(span.sum()), int(span.numel())


def score_suite(model, tok, items: list[tuple], device) -> float:
    correct = 0
    for context, continuations, gold in items:
        contexts = context if isinstance(context, list) else [context] * len(continuations)
        best, best_index = -math.inf, 0
        for index, (ctx, cont) in enumerate(zip(contexts, continuations)):
            total, count = continuation_logprob(model, tok, ctx, cont, device)
            # Length normalisation: without it the shortest option wins on every suite whose
            # options differ in length, which is most of them.
            score = total / max(count, 1)
            if score > best:
                best, best_index = score, index
        correct += int(best_index == gold)
    return 100.0 * correct / max(len(items), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a Hugging Face checkpoint directory")
    ap.add_argument("--adapter", default="", help="a LoRA adapter to score alongside the base")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suites", default=",".join(LOADERS))
    ap.add_argument("--out-dir", default="runs/general")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    suites = [name for name in args.suites.split(",") if name in LOADERS]
    loaded = {name: read_suite(name, args.rows) for name in suites}
    for name, items in loaded.items():
        print(f"{name:14s} rows={len(items)}", flush=True)

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    results = {"model": args.model, "adapter": args.adapter, "tag": args.tag,
               "rows_requested": args.rows,
               "rows": {name: len(items) for name, items in loaded.items()}, "scores": {}}

    for stage in ("base", "finetuned") if args.adapter else ("base",):
        started = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, trust_remote_code=True).to(device).eval()
        # Granite-4.0's cache asserts on layer types even for pure scoring; nothing here decodes,
        # so the cache is dead weight either way.
        model.config.use_cache = False
        if stage == "finetuned":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter).eval()
        stage_scores = {}
        for name in suites:
            stage_scores[name] = round(score_suite(model, tok, loaded[name], device), 1)
            print(f"  {stage:10s} {name:14s} {stage_scores[name]:.1f}", flush=True)
        stage_scores["seconds"] = round(time.time() - started, 1)
        results["scores"][stage] = stage_scores
        del model
        torch.cuda.empty_cache()

    if "finetuned" in results["scores"]:
        base, tuned = results["scores"]["base"], results["scores"]["finetuned"]
        results["delta"] = {name: round(tuned[name] - base[name], 1) for name in suites}
        results["mean_delta"] = round(sum(results["delta"].values()) / max(len(suites), 1), 2)

    out = Path(args.out_dir) / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results.get("delta", results["scores"]), indent=2))
    print("GENCAP_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
