#!/usr/bin/env python
"""Build a distillation corpus by relabelling the union corpus with a fine-tuned open teacher.

Weight projection across architectures transfers nothing measurable, because the donor's
coordinates mean nothing in the student's basis. Behaviour has no such problem: the teacher's
*text* is tokenizer-agnostic, so a differently-shaped, differently-tokenized open model can still
supply targets. This is sequence-level distillation, with one addition — where the teacher's call
disagrees with the gold label we keep the gold, so the student is never taught a wrong tool.

  python scripts/distill_teacher.py --teacher data/baselines/LFM2-350M \
      --adapter runs/lora/lfm2-350m --out data/distill/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from eval_suite import (accepts_system_role, chat_messages, parse_call, render_chat_prompt,
                        task_from_conversation)

from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from localagent.data.render import render_conversation_rows
from localagent.data.schema import ToolCall
from localagent.model.tokenizer import load_tokenizer
from localagent.train.stage_data import read_conversations


def relabelled(conversation, name: str, arguments: dict):
    """The conversation with its assistant call replaced."""
    messages = []
    for message in conversation.messages:
        if message.tool_calls:
            message = replace(message, tool_calls=(ToolCall(name=name,
                                                            arguments=dict(arguments)),))
        messages.append(message)
    return replace(conversation, messages=tuple(messages))


def trainable(conversation, tok, max_seq_len: int) -> bool:
    """Whether the trainer will accept the row. The teacher can emit argument keys that violate
    the tool's declared schema, and the contract refuses those outright rather than truncating."""
    try:
        render_conversation_rows(conversation, tok, prompt_contract=OPENAI_FULL_CATALOG_V1,
                                 max_seq_len=max_seq_len)
    except (ValueError, KeyError):
        return False
    return True


def teacher_outputs(base: str, adapter: str | None, prompts: list[str], batch_size: int,
                    max_new_tokens: int) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
    except TypeError:
        # Older transformers (the face-h100 overlay) spells the argument torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to("cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()

    outputs, started = [], time.time()
    for index in range(0, len(prompts), batch_size):
        chunk = prompts[index: index + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=1536).to("cuda")
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        for row, source in zip(generated, encoded["input_ids"]):
            outputs.append(tokenizer.decode(row[source.shape[0]:], skip_special_tokens=True))
        if index % (batch_size * 20) == 0:
            done = index + len(chunk)
            rate = done / max(time.time() - started, 1e-6)
            print(f"[teacher] {done}/{len(prompts)} rows  {rate:.1f} rows/s", flush=True)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--source", default="data/merged-v2/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", default="data/tokenizer-h100-16k.json")
    args = ap.parse_args()

    # Pair each conversation with its own task in one pass: deriving the two lists separately
    # would let a dropped row shift the alignment and attach teacher outputs to the wrong prompts.
    usable, tasks = [], []
    for conversation in read_conversations(Path(args.source)):
        task = task_from_conversation(conversation)
        if task is None:
            continue
        usable.append(conversation)
        tasks.append(task)
        if len(tasks) >= args.rows:
            break
    print(f"paired conversations/tasks = {len(usable)}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    supports_system = accepts_system_role(tokenizer)
    prompts = [render_chat_prompt(tokenizer, chat_messages(task, supports_system))
               for task in tasks]
    generations = teacher_outputs(args.teacher, args.adapter, prompts, args.batch_size,
                                  args.max_new_tokens)

    tok = load_tokenizer("bpe", args.tokenizer)
    kept_teacher = kept_gold = dropped = 0
    rows = []
    for conversation, task, text in zip(usable, tasks, generations):
        parsed = parse_call(text)
        candidate = None
        if parsed is not None and parsed[0] == task.gold_name:
            candidate = relabelled(conversation, parsed[0], parsed[1])
            if not trainable(candidate, tok, args.max_seq_len):
                candidate = None
        if candidate is not None:
            kept_teacher += 1
        else:
            candidate = relabelled(conversation, task.gold_name, task.gold_arguments)
            if not trainable(candidate, tok, args.max_seq_len):
                dropped += 1
                continue
            kept_gold += 1
        rows.append(candidate)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), default=str, sort_keys=True) + "\n")
    manifest = {"teacher": args.teacher, "adapter": args.adapter, "source": args.source,
                "rows": len(rows), "kept_teacher_call": kept_teacher, "kept_gold_call": kept_gold,
                "dropped_unrenderable": dropped, "max_new_tokens": args.max_new_tokens}
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("DISTILL_CORPUS_DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
