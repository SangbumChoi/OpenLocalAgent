#!/usr/bin/env python
"""Published-metric scoring for a trained arm, so results are comparable outside this repository.

Teacher-forced token accuracy compares arms to each other but matches no published number. This
generates the assistant turn and scores it the way each benchmark's own paper does:

  AndroidControl   Type Match, Grounding Accuracy (14% screen-width radius), Step Success Rate
                   (Li et al., 2024)
  ToolACE / BFCL   AST-style function-call match: name, argument names, argument values
                   (Patil et al., Berkeley Function Calling Leaderboard)
  AgentNet         action-type match and full-step match over desktop trajectories

  python scripts/benchmark_metrics.py --checkpoint runs/region/data-union/model.pt \
      --out runs/region/data-union/benchmark_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from localagent.inference.generate import generate
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer
from localagent.train.stage_data import read_conversations

PUBLIC = Path("data/public")
SUITES = {
    "androidcontrol": PUBLIC / "androidcontrol-test.jsonl",
    "toolace": PUBLIC / "toolace-eval.jsonl",
    "agentnet": PUBLIC / "agentnet-eval.jsonl",
}
# AndroidControl screenshots in the public mirror are 1080 px wide; the benchmark counts a tap as
# grounded within 14% of screen width of the gold coordinate.
SCREEN_WIDTH = 1080
GROUNDING_RADIUS = 0.14 * SCREEN_WIDTH


def gold_call(conversation) -> tuple[str, dict] | None:
    for message in conversation.messages:
        role = getattr(message.role, "value", message.role)
        if role == "assistant" and message.tool_calls:
            call = message.tool_calls[0]
            return call.name, dict(call.arguments)
    return None


def prompt_for(conversation, budget: int) -> str:
    """User turns up to the first assistant decision, in the byte model's own framing.

    Long observations are truncated from the left: the instruction and the assistant marker sit at
    the end of the prompt, and that tail is what the decision depends on.
    """
    parts = []
    for message in conversation.messages:
        role = getattr(message.role, "value", message.role)
        if role == "assistant":
            break
        if role in ("user", "system") and message.content:
            parts.append(message.content)
    body = " ".join(parts)
    room = budget - len(USER) - len(ASSISTANT)
    if room > 0 and len(body) > room:
        body = body[-room:]
    return f"{USER}{body}{ASSISTANT}"


def parse_call(text: str) -> tuple[str, dict] | None:
    match = re.search(r'\{.*?"name"\s*:\s*"([^"]+)".*?\}', text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return match.group(1), {}
    if not isinstance(payload, dict):
        return match.group(1), {}
    return str(payload.get("name", match.group(1))), dict(payload.get("arguments") or {})


def arguments_match(gold: dict, predicted: dict) -> bool:
    if set(gold) != set(predicted):
        return False
    return all(str(gold[key]).strip() == str(predicted[key]).strip() for key in gold)


def coordinates_within_radius(gold: dict, predicted: dict) -> bool | None:
    if not {"x", "y"} <= set(gold):
        return None
    try:
        distance = math.dist((float(gold["x"]), float(gold["y"])),
                             (float(predicted.get("x", 1e9)), float(predicted.get("y", 1e9))))
    except (TypeError, ValueError):
        return False
    return distance <= GROUNDING_RADIUS


def score_suite(model, tok, rows, device, max_new_tokens: int, budget: int) -> dict[str, float]:
    counts = {"rows": 0, "parsed": 0, "type_match": 0, "grounded": 0, "grounded_eligible": 0,
              "step_success": 0}
    for conversation in rows:
        gold = gold_call(conversation)
        if gold is None:
            continue
        gold_name, gold_args = gold
        counts["rows"] += 1
        generated = generate(model, tok, prompt_for(conversation, budget),
                             max_new_tokens=max_new_tokens)
        text = generated[0] if isinstance(generated, tuple) else generated
        predicted = parse_call(text)
        if predicted is None:
            continue
        counts["parsed"] += 1
        predicted_name, predicted_args = predicted
        type_ok = predicted_name == gold_name
        counts["type_match"] += int(type_ok)
        grounded = coordinates_within_radius(gold_args, predicted_args)
        if grounded is not None:
            counts["grounded_eligible"] += 1
            counts["grounded"] += int(type_ok and grounded)
        counts["step_success"] += int(type_ok and arguments_match(gold_args, predicted_args))
    total = max(counts["rows"], 1)
    report = {
        "rows": counts["rows"],
        "parse_rate": counts["parsed"] / total,
        "type_match": counts["type_match"] / total,
        "step_success_rate": counts["step_success"] / total,
    }
    if counts["grounded_eligible"]:
        report["grounding_accuracy"] = counts["grounded"] / counts["grounded_eligible"]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["cfg"])
    model = LocalAgentLM(cfg).to(args.device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tok = load_tokenizer("byte")

    report = {"checkpoint": args.checkpoint, "rows_per_suite": args.rows,
              "grounding_radius_px": GROUNDING_RADIUS, "suites": {}}
    for name, path in SUITES.items():
        if not path.exists():
            continue
        rows = read_conversations(path)[: args.rows]
        budget = cfg.max_seq_len - args.max_new_tokens - 16
        report["suites"][name] = score_suite(model, tok, rows, args.device,
                                            args.max_new_tokens, budget)
        line = " ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in report["suites"][name].items())
        print(f"{name}: {line}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("BENCHMARK_METRICS_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
