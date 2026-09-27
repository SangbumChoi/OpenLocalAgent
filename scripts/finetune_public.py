#!/usr/bin/env python
"""Fine-tune a public instruction model on the union agent corpus with one shared recipe.

Every model gets the identical treatment — same LoRA shape, same optimizer, same step budget, same
rows, and the prompt builder the evaluator uses — so the fine-tuned column compares models rather
than recipes. Loss is taken only on the answer span, because the prompt carries the tool catalog
and training the model to reproduce it would waste the whole budget.

  python scripts/finetune_public.py --base data/baselines/SmolLM2-360M-Instruct \
      --out runs/lora/smollm2-360m --steps 600
"""

from __future__ import annotations

import argparse
import re
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from eval_suite import accepts_system_role, build_tasks, chat_messages, render_chat_prompt

TRAIN_ROWS = Path("data/merged-v2/train.jsonl")


def answer_text(task) -> str:
    """The target string: exactly the shape the shared parser reads back."""
    return json.dumps({"name": task.gold_name, "arguments": task.gold_arguments})


def build_examples(tokenizer, tasks, supports_system: bool, max_length: int):
    """Tokenized (input_ids, labels) with the prompt masked out of the loss."""
    examples = []
    for task in tasks:
        prompt = render_chat_prompt(tokenizer, chat_messages(task, supports_system))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text(task) + (tokenizer.eos_token or ""),
                               add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= max_length:
            continue
        # Keep the tail of the prompt: the request sits at the end, after the catalog.
        # (Guard the zero case: x[-0:] is the whole list, not the empty one.)
        room = max(1, max_length - len(answer_ids))
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        examples.append((input_ids, labels))
    return examples


def collate(batch, pad_id: int):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, mask = [], [], []
    for ids, label in batch:
        padding = width - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        labels.append(label + [-100] * padding)
        mask.append([1] * len(ids) + [0] * padding)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(mask))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    # Restrict adaptation to a depth band. The causal test of "which layers hold agentic ability"
    # is to let only those layers move and see how much of the full-recipe gain survives.
    ap.add_argument("--layers", default="", help="comma list or a-b range of layer indices; "
                                                 "empty adapts every layer")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False

    targets = sorted({name.split(".")[-1] for name, module in model.named_modules()
                      if isinstance(module, torch.nn.Linear)
                      and name.split(".")[-1] not in ("lm_head", "score")})
    if args.layers:
        # peft accepts fully-qualified module names, so the band is expressed by naming every
        # target inside the chosen layers instead of by suffix.
        chosen: set[int] = set()
        for span in args.layers.split(","):
            if "-" in span:
                lo, hi = span.split("-")
                chosen |= set(range(int(lo), int(hi) + 1))
            elif span.strip():
                chosen.add(int(span))
        layer_of = re.compile(r"\.(?:layers|blocks|h)\.(\d+)\.")
        named = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in targets
                 and (found := layer_of.search(name)) and int(found.group(1)) in chosen]
        if not named:
            raise SystemExit(f"--layers {args.layers} matched nothing")
        targets = named
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    # Keep the adapter in fp32 while the frozen base stays bf16: at this learning rate a pure
    # bf16 adapter drives the loss to NaN on some families, which reads as a dead model rather
    # than an unstable optimiser.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"targets={targets} trainable={trainable/1e6:.2f}M", flush=True)

    tasks = build_tasks(TRAIN_ROWS, args.rows)
    examples = build_examples(tokenizer, tasks, accepts_system_role(tokenizer), args.max_length)
    random.Random(args.seed).shuffle(examples)
    print(f"examples={len(examples)}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    warmup = max(1, args.steps // 20)

    def learning_rate_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    model.train()
    started, cursor, skipped = time.time(), 0, 0
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(examples):
            cursor = 0
        batch = examples[cursor: cursor + args.batch_size]
        cursor += args.batch_size
        input_ids, labels, mask = (tensor.to("cuda") for tensor in collate(
            batch, tokenizer.pad_token_id))
        loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        # Guard the gradient, not just the loss: in bf16 a finite loss can still yield an
        # inf gradient, and one optimizer step on that turns the adapter into NaN permanently.
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(total_norm):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            continue
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == 1:
            print(f"[lora] step {step}/{args.steps} loss {loss.item():.4f} "
                  f"{time.time() - started:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"non-finite steps skipped: {skipped}", flush=True)
    (out / "recipe.json").write_text(json.dumps({
        "non_finite_steps_skipped": skipped,
        "base": args.base, "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "rank": args.rank, "max_length": args.max_length,
        "examples": len(examples), "trainable_parameters": trainable,
        "target_modules": targets, "seed": args.seed, "layers": args.layers,
        "corpus": str(TRAIN_ROWS)}, indent=2) + "\n")
    print("FINETUNE_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
