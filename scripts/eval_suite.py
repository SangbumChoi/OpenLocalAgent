#!/usr/bin/env python
"""One evaluation process for every model, so a sub-30M byte agent and a sub-1B instruct model
are scored on identical tasks with identical metrics.

A task is model-agnostic: the observation text, the tool catalog, and the gold call. Each adapter
renders that task in its own native format and returns a predicted call; scoring is shared.

Metrics follow the published definitions:
  type match          predicted function name equals gold (AndroidControl Type Match)
  step success rate   name and every argument correct (AndroidControl SR / BFCL AST exact)
  grounding accuracy  tap within 14% of screen width of gold (AndroidControl GR)
  parse rate          a syntactically valid call was produced at all

  python scripts/eval_suite.py --model localagent:runs/region/data-union/model.pt --rows 200
  python scripts/eval_suite.py --model hf:data/baselines/SmolLM2-360M-Instruct --rows 200
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from localagent.train.stage_data import read_conversations

PUBLIC = Path("data/public")
SUITES = {
    "androidcontrol": PUBLIC / "androidcontrol-test.jsonl",
    # Held-out by rendered-prompt hash from the pinned Mind2Web *train* file, since repository
    # policy keeps the official test split out of the checkout. Web element selection, in
    # distribution — not an official Mind2Web score.
    "mind2web": Path("data/merged-v2/eval-mind2web.jsonl"),
    "toolace": PUBLIC / "toolace-eval.jsonl",
    "xlam": PUBLIC / "xlam-test.jsonl",
    # Evaluation-only by repository policy: BFCL is never merged into a training split.
    "bfcl": PUBLIC / "bfcl-eval.jsonl",
    "agentnet": PUBLIC / "agentnet-eval.jsonl",
    # Evaluation-only by the same policy. First action of the released DFS trajectory, not a
    # ToolEval pass rate — see scripts/normalize_toolbench.py for the claim boundary.
    "toolbench": PUBLIC / "toolbench-eval.jsonl",
}
SCREEN_WIDTH = 1080
GROUNDING_RADIUS = 0.14 * SCREEN_WIDTH


@dataclass(frozen=True)
class Task:
    observation: str
    tools: tuple[dict, ...]
    gold_name: str
    gold_arguments: dict


def task_from_conversation(conversation) -> Task | None:
    """One conversation as a scoreable task, or None if it carries no gold call.

    Exposed so a relabelling pass can pair a conversation with its own task rather than
    re-deriving the list and risking a different order.
    """
    observation, gold = [], None
    for message in conversation.messages:
        role = getattr(message.role, "value", message.role)
        if role == "assistant" and message.tool_calls:
            call = message.tool_calls[0]
            gold = (call.name, dict(call.arguments))
            break
        if message.content:
            observation.append(message.content)
        if getattr(message, "tool_response", None):
            observation.append(str(message.tool_response))
    if gold is None:
        return None
    catalog = tuple({"name": tool.name,
                     "description": getattr(tool, "description", "") or "",
                     "parameters": getattr(tool, "parameters", {}) or {}}
                    for tool in (conversation.tools or []))
    return Task(" ".join(observation), catalog, gold[0], gold[1])


def build_tasks(path: Path, limit: int) -> list[Task]:
    tasks = []
    for conversation in read_conversations(path)[: limit * 2]:
        task = task_from_conversation(conversation)
        if task is None:
            continue
        tasks.append(task)
        if len(tasks) >= limit:
            break
    return _with_suite_catalog(tasks)


def _with_suite_catalog(tasks: list[Task]) -> list[Task]:
    """Give rows that carry no catalog the suite's own action space.

    Some sources (AgentNet) record the action but not the tool list. Without a catalog neither a
    retriever nor a prompted model can be asked to choose, so every model would score zero for the
    same uninformative reason. The suite-level catalog is derived from the gold calls themselves and
    is identical for every model.
    """
    if all(task.tools for task in tasks):
        return tasks
    schema: dict[str, dict] = {}
    for task in tasks:
        properties = schema.setdefault(task.gold_name, {})
        for key in task.gold_arguments:
            properties[key] = {"type": "string"}
    catalog = tuple({"name": name,
                     "description": name.replace("_", " "),
                     "parameters": {"type": "object", "properties": properties,
                                    "required": sorted(properties)}}
                    for name, properties in sorted(schema.items()))
    return [task if task.tools else Task(task.observation, catalog, task.gold_name,
                                         task.gold_arguments)
            for task in tasks]


def parse_pythonic_call(text: str) -> tuple[str, dict] | None:
    """A `[name(arg="value", n=1)]` call, the format the LFM2 chat template teaches.

    Scored alongside JSON because a model that names the right tool in its own house style has
    answered the question; rejecting the spelling would measure the parser, not the agent.
    """
    match = re.search(r"\[?\s*([A-Za-z_][\w .-]*)\(([^()]*)\)\s*\]?", text)
    if not match:
        return None
    arguments = {}
    for pair in re.finditer(r"([A-Za-z_]\w*)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]+)", match.group(2)):
        raw = pair.group(2).strip()
        try:
            arguments[pair.group(1)] = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            arguments[pair.group(1)] = raw.strip("\"'")
    return match.group(1).strip(), arguments


def parse_call(text: str) -> tuple[str, dict] | None:
    """First JSON object carrying a function name, in any of the emitted spellings."""
    for candidate in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("function") or payload.get("tool")
        if isinstance(name, str) and name:
            arguments = payload.get("arguments") or payload.get("parameters") or payload.get("args")
            return name, dict(arguments) if isinstance(arguments, dict) else {}
    match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    return (match.group(1), {}) if match else parse_pythonic_call(text)


class LocalAgentAdapter:
    """The repository's byte-level checkpoints, rendered exactly as they were trained."""

    kind = "localagent"

    def __init__(self, checkpoint: str, device: str):
        from localagent.model import LocalAgentLM, ModelConfig
        from localagent.model.tokenizer import load_tokenizer

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ModelConfig(**payload["cfg"])
        self.model = LocalAgentLM(self.cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.tok = load_tokenizer("byte")
        self.device = device
        self.name = Path(checkpoint).parent.name
        self.parameters = self.model.num_params()

    # One byte is one token here, while a BPE model spends roughly one token per four characters.
    # The budget is scaled so every family is allowed the same amount of *text*, not the same
    # number of its own tokens; a tool call is ~120 bytes and would otherwise be truncated.
    BYTES_PER_BPE_TOKEN = 4

    def predict(self, task: Task, max_new_tokens: int) -> str:
        from localagent.inference.generate import generate
        from localagent.model.tokenizer import ASSISTANT, USER

        max_new_tokens = max_new_tokens * self.BYTES_PER_BPE_TOKEN

        # Truncate on tokens, not characters: multi-byte UTF-8 makes the byte length longer than
        # the string length, and the model's context is counted in tokens.
        budget = self.cfg.max_seq_len - max_new_tokens - len(self.tok.encode(USER + ASSISTANT)) - 8
        body_ids = self.tok.encode(task.observation)
        if len(body_ids) > budget:
            body_ids = body_ids[-budget:]
        prompt = f"{USER}{self.tok.decode(body_ids)}{ASSISTANT}"
        generated = generate(self.model, self.tok, prompt, max_new_tokens=max_new_tokens)
        text = generated[0] if isinstance(generated, tuple) else generated
        return text[len(prompt):] if text.startswith(prompt) else text


SYSTEM_PROMPT = ("You control a device. Choose exactly one function from the catalog and answer "
                 'with only one JSON object: {"name": ..., "arguments": {...}}. No prose.')


def chat_messages(task: Task, supports_system: bool = True) -> list[dict]:
    """The task as chat turns. Shared with the fine-tuner so training and scoring cannot drift."""
    catalog = json.dumps([{"name": tool["name"], "parameters": tool["parameters"]}
                          for tool in task.tools])[:2000]
    user = f"Catalog: {catalog}\nRequest: {task.observation[-2000:]}"
    if not supports_system:
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def accepts_system_role(tokenizer) -> bool:
    """Some chat templates raise on a system turn; those models get it folded into the user turn
    instead, so a template convention does not read as a capability difference."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return False
    return True


def render_chat_prompt(tokenizer, messages: list[dict]) -> str:
    try:
        # Reasoning models spend the whole generation budget thinking and never reach the call,
        # so ask for the non-thinking path where the template offers one.
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (ValueError, AttributeError):
        return "\n\n".join(message["content"] for message in messages) + "\n"


class HuggingFaceAdapter:
    """Any causal instruct model under the same task definition, using its own chat template."""

    kind = "hf"

    def __init__(self, path: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # bf16 on the GPU: greedy decoding over a 64-token budget is insensitive to it, and fp32
        # would put a nine-model sweep out of reach. The CPU pass stays fp32.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype).to(device).eval()
        self.device = device
        self.name = Path(path).name
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.supports_system = self._accepts_system_role()

    def _accepts_system_role(self) -> bool:
        return accepts_system_role(self.tokenizer)

    def predict(self, task: Task, max_new_tokens: int) -> str:
        prompt = render_chat_prompt(
            self.tokenizer, chat_messages(task, self.supports_system))
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=1536).to(self.device)
        with torch.no_grad():
            try:
                output = self.model.generate(**encoded, max_new_tokens=max_new_tokens,
                                             do_sample=False,
                                             pad_token_id=self.tokenizer.eos_token_id)
            except ValueError:
                # Some hybrid-attention families raise from the KV cache on this transformers
                # version; decoding without the cache is slower but gives the same tokens.
                output = self.model.generate(**encoded, max_new_tokens=max_new_tokens,
                                             do_sample=False, use_cache=False,
                                             pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][encoded["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


class LoraAdapter(HuggingFaceAdapter):
    """A base model with a LoRA adapter merged in, scored exactly like the released model."""

    kind = "lora"

    def __init__(self, spec: str, device: str):
        base, _, adapter = spec.partition("|")
        super().__init__(base, device)
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload().eval()
        self.name = Path(adapter).name
        self.parameters = sum(p.numel() for p in self.model.parameters())


class DispatchAdapter:
    """The deployed path: retrieve a tool from the task's own catalog, then ground its arguments.

    No language-model generation is involved, which is the point — this is what actually runs on a
    CPU-only device, and it is scored on exactly the same tasks as every generative model.
    """

    kind = "dispatch"

    def __init__(self, location: str, device: str):
        self.device = device
        self.name = location or "retrieve+ground"
        self.parameters = 0
        self._cache: dict[tuple[str, ...], object] = {}

    def _caller(self, task: Task):
        from localagent.agent.caller import ToolCaller
        from localagent.agent.toolset import ToolSpec

        key = tuple(sorted(tool["name"] for tool in task.tools))
        if key not in self._cache:
            specs = [ToolSpec(name=tool["name"], description=tool["description"] or tool["name"],
                              parameters=tool["parameters"] or {"type": "object", "properties": {}})
                     for tool in task.tools]
            self._cache[key] = ToolCaller(specs) if specs else None
        return self._cache[key]

    def predict(self, task: Task, max_new_tokens: int) -> str:
        caller = self._caller(task)
        if caller is None:
            return ""
        call = caller.call(task.observation)
        if call is None:
            return ""
        return json.dumps({"name": call.name, "arguments": dict(call.arguments)})


class CatalogAdapter:
    """A BPE LocalAgent checkpoint trained with the catalog in its prompt.

    Unlike the byte checkpoints, this model reads its action space from the request, so it can be
    asked about tools that were never in its training set — the same question the instruct
    baselines are asked, in the model's own contract.
    """

    kind = "catalog"

    def __init__(self, checkpoint: str, device: str):
        from localagent.model import LocalAgentLM, ModelConfig
        from localagent.model.tokenizer import load_tokenizer

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("cfg") or payload.get("config")
        self.cfg = ModelConfig(**config)
        self.model = LocalAgentLM(self.cfg).to(device)
        state = payload.get("state_dict") or payload.get("model")
        self.model.load_state_dict(state)
        self.model.eval()
        self.tok = load_tokenizer("bpe", "data/tokenizer-h100-16k.json")
        self.device = device
        self.name = Path(checkpoint).parent.name
        self.parameters = self.model.num_params()

    def predict(self, task: Task, max_new_tokens: int) -> str:
        from localagent.data.prompt_contract import render_agent_decode_prompt
        from localagent.data.schema import Message, Role, ToolSpec
        from localagent.inference.generate import generate

        tools = [ToolSpec(name=tool["name"], description=tool["description"] or tool["name"],
                          parameters=tool["parameters"] or {"type": "object", "properties": {}})
                 for tool in task.tools]
        messages = [Message(role=Role.user, content=task.observation[:4000])]
        try:
            prompt = render_agent_decode_prompt(messages, tools)
        except (ValueError, KeyError):
            return ""
        ids = self.tok.encode(prompt)
        room = self.cfg.max_seq_len - max_new_tokens - 4
        if len(ids) > room:
            return ""
        generated = generate(self.model, self.tok, prompt, max_new_tokens=max_new_tokens)
        text = generated[0] if isinstance(generated, tuple) else generated
        return text[len(prompt):] if text.startswith(prompt) else text


# Some sources pack the click point into one string, e.g. "button=left;x=0.018;y=0.508".
PACKED_POINT = re.compile(r"x=(-?[\d.]+).*?y=(-?[\d.]+)")


def click_point(arguments: dict) -> tuple[float, float, float] | None:
    """The click point and the extent of the coordinate space, however the source spells it.

    Returns (x, y, extent) where extent is the screen width for pixel coordinates and 1.0 for
    normalised ones, so one tolerance rule covers both.
    """
    if {"x", "y"} <= set(arguments):
        try:
            return float(arguments["x"]), float(arguments["y"]), float(SCREEN_WIDTH)
        except (TypeError, ValueError):
            return None
    for value in arguments.values():
        if isinstance(value, str):
            found = PACKED_POINT.search(value)
            if found:
                try:
                    x, y = float(found.group(1)), float(found.group(2))
                except ValueError:
                    return None
                # Normalised sources write 0..1; a pixel source would exceed that.
                return x, y, 1.0 if max(abs(x), abs(y)) <= 1.0 else float(SCREEN_WIDTH)
    return None


def arguments_match(gold: dict, predicted: dict) -> bool:
    """Exact match, except that a click point counts if it lands within the grounding radius.

    Demanding a byte-identical coordinate would make step success unreachable on any suite that
    writes six decimal places, which is not what the published metric means: AndroidControl counts
    a tap correct within 14% of screen width, and the same rule is applied wherever a click point
    appears, however the source encodes it.
    """
    if set(gold) != set(predicted):
        return False
    gold_point, predicted_point = click_point(gold), click_point(predicted)
    if gold_point and predicted_point:
        radius = 0.14 * gold_point[2]
        if math.dist(gold_point[:2], predicted_point[:2]) > radius:
            return False
        # Everything that is not the coordinate string still has to match exactly.
        return all(str(gold[key]).strip() == str(predicted[key]).strip()
                   for key in gold if not PACKED_POINT.search(str(gold[key]))
                   and key not in ("x", "y"))
    return all(str(gold[key]).strip() == str(predicted[key]).strip() for key in gold)


def grounded(gold: dict, predicted: dict) -> bool | None:
    gold_point = click_point(gold)
    if gold_point is None:
        return None
    predicted_point = click_point(predicted)
    if predicted_point is None:
        return False
    return math.dist(gold_point[:2], predicted_point[:2]) <= 0.14 * gold_point[2]


def score(adapter, tasks: list[Task], max_new_tokens: int) -> dict[str, float]:
    counters = {"parsed": 0, "type_match": 0, "step_success": 0, "grounded": 0, "eligible": 0}
    started = time.time()
    for task in tasks:
        prediction = parse_call(adapter.predict(task, max_new_tokens))
        if prediction is None:
            continue
        counters["parsed"] += 1
        name, arguments = prediction
        type_ok = name == task.gold_name
        counters["type_match"] += int(type_ok)
        hit = grounded(task.gold_arguments, arguments)
        if hit is not None:
            counters["eligible"] += 1
            counters["grounded"] += int(type_ok and hit)
        counters["step_success"] += int(type_ok and arguments_match(task.gold_arguments, arguments))
    total = max(len(tasks), 1)
    report = {"rows": len(tasks),
              "parse_rate": counters["parsed"] / total,
              "type_match": counters["type_match"] / total,
              "step_success_rate": counters["step_success"] / total,
              "seconds_per_row": (time.time() - started) / total}
    if counters["eligible"]:
        report["grounding_accuracy"] = counters["grounded"] / counters["eligible"]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="localagent:<ckpt> | hf:<path> | lora:<base>|<adapter> | catalog:<ckpt>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suites", help="comma-separated subset; merged into an existing --out file")
    args = ap.parse_args()

    kind, _, location = args.model.partition(":")
    adapters = {"localagent": LocalAgentAdapter, "hf": HuggingFaceAdapter,
                "lora": LoraAdapter, "dispatch": DispatchAdapter, "catalog": CatalogAdapter}
    if kind not in adapters:
        raise SystemExit(f"unknown model kind {kind!r}; expected one of {sorted(adapters)}")
    adapter = adapters[kind](location, args.device)

    report = {"model": adapter.name, "kind": adapter.kind, "location": location,
              "parameters": adapter.parameters, "rows_per_suite": args.rows,
              "max_new_tokens": args.max_new_tokens, "suites": {}}
    wanted = set(args.suites.split(",")) if args.suites else None
    if wanted and Path(args.out).exists():
        # Re-scoring one suite updates that block in place instead of discarding the others.
        report["suites"] = json.loads(Path(args.out).read_text()).get("suites", {})
    for suite, path in SUITES.items():
        if not path.exists() or (wanted and suite not in wanted):
            continue
        tasks = build_tasks(path, args.rows)
        report["suites"][suite] = score(adapter, tasks, args.max_new_tokens)
        line = " ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in report["suites"][suite].items())
        print(f"{adapter.name} {suite}: {line}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("EVAL_SUITE_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
