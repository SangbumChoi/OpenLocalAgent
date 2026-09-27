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

  openlocalagent-eval-suite --model openlocalagent:results/runs/la-93m/model.pt --out results/la-93m.json
  openlocalagent-eval-suite --model hf:data/baselines/SmolLM2-360M-Instruct --out results/smol.json
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from openlocalagent.train.stage_data import read_conversations

PUBLIC = Path("data/public")

#: Set by --allow-tokenizer-mismatch. Off by default: a vocabulary mismatch must stop a run
#: rather than quietly become a score.
ALLOW_TOKENIZER_MISMATCH = False
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
    # Evaluation-only by the same policy. Single-turn projection of a stateful benchmark; the
    # 17 rows are ToolSandbox's single-tool scenarios with an unambiguous allow-listed gold —
    # see scripts/normalize_toolsandbox.py for the claim boundary.
    "toolsandbox": PUBLIC / "toolsandbox-eval.jsonl",
    # Evaluation-only. First action of the reference trajectory against the task's enabled MCP
    # tools (name-only catalogs; MCP-Atlas ships no parameter schemas) — 495 rows, CC-BY-4.0.
    "mcpatlas": PUBLIC / "mcpatlas-eval.jsonl",
    # The dataset authors' own eval split (FunctionGemma's Android-tools distribution, CC-BY-4.0);
    # 961 rows, mean catalog 7.0. The train split is a legitimate Q2 import candidate, kept out
    # of every training run until that arm is deliberately built.
    "mobileactions": PUBLIC / "mobileactions-eval.jsonl",
}
# Two derived views of xLAM that ask whether its result depends on the tool names themselves.
# xLAM's evaluation names overlap its training names almost completely, so a gain there is
# consistent both with the model having learned the catalog and with it having memorised a
# name-to-function mapping. These separate the two by rewriting only the names, per row, leaving
# every description, parameter schema and question untouched, and rewriting the gold answer the
# same way so the task stays solvable.
#
#   xlam_opaque    every name becomes tool_0, tool_1, ... in catalog order. A meaningful name
#                  carries no information any more, so what survives is what the description and
#                  the schema support.
#   xlam_shuffled  names are cyclically rotated among the row's tools, so each name is attached to
#                  a different tool's description and schema, and the gold answer is whatever name
#                  now sits on the correct function. A model that reads the catalog is unaffected;
#                  one that goes by the name alone is actively misled.
#
# Both are deterministic given the row, take no seed, and are derived from the same pinned
# xlam-test.jsonl, so any model can be re-scored on them at any time.
RENAME_VIEWS = ("xlam_opaque", "xlam_shuffled")

# Suites that are real benchmarks but deliberately outside the headline ten, because adding one
# would silently change what "the ten-benchmark average" means and break comparability with every
# receipt already in results/evalsuite-full. They run only when named, like the rename views.
#
#   tau2  tau2-bench (sierra-research/tau2-bench, MIT), projected to the first AGENT action of the
#         reference solution - NOT a tau2-bench Pass^k, which needs a user simulator, a mutable
#         database and an LLM judge. Evaluation-only and in no training mixture, so its score is
#         pure transfer: airline, retail, telecom and banking policies appear nowhere in training.
#         Built by scripts/normalize_tau2.py.
SUPPLEMENTAL_SUITES = {
    "tau2": PUBLIC / "tau2-eval.jsonl",
}


def _rename_task(task: "Task", view: str) -> "Task":
    names = [tool["name"] for tool in task.tools]
    if not names or task.gold_name not in names:
        return task
    if view == "xlam_opaque":
        mapping = {name: f"tool_{index}" for index, name in enumerate(names)}
    elif view == "xlam_shuffled":
        if len(names) < 2:
            return task
        rotated = names[1:] + names[:1]
        mapping = dict(zip(names, rotated))
    else:
        raise ValueError(f"unknown rename view {view!r}")
    tools = tuple({**tool, "name": mapping[tool["name"]]} for tool in task.tools)
    return Task(task.observation, tools, mapping[task.gold_name], task.gold_arguments)


def build_rename_tasks(view: str, limit: int) -> list["Task"]:
    """xLAM rows with only the tool names rewritten, gold answer included."""
    return [_rename_task(task, view) for task in build_tasks(SUITES["xlam"], limit)]


SCREEN_WIDTH = 1080
GROUNDING_RADIUS = 0.14 * SCREEN_WIDTH


@dataclass(frozen=True)
class TokenizerLineage:
    """Whether a checkpoint's vocabulary is the one this tokenizer file encodes.

    Training refuses to chain a stage onto a checkpoint whose tokenizer moved; evaluation loaded a
    file chosen by vocab_size and scored whatever came out. That asymmetry means training fails
    loudly and scoring fails silently - the worse half, because a silent failure becomes a
    published number. It has happened once already in the other direction: 64k checkpoints decoded
    under the 16k table scored 0.0 across every suite, and the cause was found by hand.

    Then packing a new corpus without --reuse-tokenizer retrained the shared BPE in place, and the
    original had no other copy on the box. Every checkpoint from that ladder now names a
    vocabulary that no longer exists. This type is what stops that becoming a receipt.
    """

    checkpoint_sha256: str
    tokenizer_sha256: str
    tokenizer_path: str

    @property
    def matches(self) -> bool:
        return self.checkpoint_sha256 == self.tokenizer_sha256

    @property
    def complaint(self) -> str:
        return (
            f"checkpoint was trained against tokenizer sha256 {self.checkpoint_sha256[:16]}... "
            f"but {self.tokenizer_path} is {self.tokenizer_sha256[:16]}.... Decoding under the "
            "wrong vocabulary produces well-formed nonsense, so this run would report a number "
            "rather than an error. Pass --allow-tokenizer-mismatch to score anyway; the receipt "
            "then records that it is not comparable with any other."
        )

    def as_receipt_note(self) -> dict:
        return {"checkpoint_tokenizer_sha256": self.checkpoint_sha256,
                "loaded_tokenizer_sha256": self.tokenizer_sha256,
                "loaded_tokenizer_path": self.tokenizer_path,
                "warning": "scored under a tokenizer the checkpoint was not trained with; "
                           "not comparable with any other receipt"}


def tokenizer_lineage(payload, tokenizer_path: str) -> TokenizerLineage | None:
    """Compare a checkpoint's recorded tokenizer identity with the file about to be loaded.

    None when the checkpoint records no identity: older artifacts predate the metadata, and
    refusing to score them would retire results that were valid when produced.
    """
    import hashlib

    from openlocalagent.train.stage_data import checkpoint_tokenizer_sha256

    try:
        recorded = checkpoint_tokenizer_sha256(payload)
    except Exception:
        return None
    path = Path(tokenizer_path)
    if not recorded or not path.is_file():
        return None
    return TokenizerLineage(recorded, hashlib.sha256(path.read_bytes()).hexdigest(),
                            str(tokenizer_path))


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

    kind = "openlocalagent"

    def __init__(self, checkpoint: str, device: str):
        from openlocalagent.model import LocalAgentLM, ModelConfig
        from openlocalagent.model.tokenizer import load_tokenizer

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

    def _prompt_for(self, task: Task, max_new_tokens: int) -> tuple[str, int]:
        """Exactly the prompt predict() builds, so the batched path cannot drift from it."""
        from openlocalagent.model.tokenizer import ASSISTANT, USER

        scaled = max_new_tokens * self.BYTES_PER_BPE_TOKEN
        budget = self.cfg.max_seq_len - scaled - len(self.tok.encode(USER + ASSISTANT)) - 8
        body_ids = self.tok.encode(task.observation)
        if len(body_ids) > budget:
            body_ids = body_ids[-budget:]
        return f"{USER}{self.tok.decode(body_ids)}{ASSISTANT}", scaled

    def predict_many(self, tasks: list[Task], max_new_tokens: int) -> list[str]:
        if len(tasks) == 1:
            return [self.predict(tasks[0], max_new_tokens)]
        built = [self._prompt_for(task, max_new_tokens) for task in tasks]
        return _batched_catalog_generate(self.model, self.tok, [b[0] for b in built], built[0][1])

    def predict(self, task: Task, max_new_tokens: int) -> str:
        from openlocalagent.inference.generate import generate
        from openlocalagent.model.tokenizer import ASSISTANT, USER

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



@torch.no_grad()
def _batched_catalog_generate(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy decode for the custom architecture, batched by EXACT prompt length.

    RoPE here broadcasts one absolute-position vector across the batch (apply_rope slices cos/sin
    to positions, then indexes [None, None]), so left padding would hand short rows the wrong
    positions. Grouping rows of identical token length needs no padding at all, which keeps every
    row's positions and attention exactly what they are one-at-a-time.
    """
    encoded = [tok.encode(prompt) for prompt in prompts]
    by_length: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_length.setdefault(len(ids), []).append(index)
    device = next(model.parameters()).device
    out: list[str] = [""] * len(prompts)
    for length, indices in by_length.items():
        rows = torch.tensor([encoded[i] for i in indices], dtype=torch.long, device=device)
        caches = [None] * model.n_cache_slots()
        logits, _, caches = model(rows, pos=0, caches=caches)
        pos = length
        collected: list[list[int]] = [[] for _ in indices]
        finished = [False] * len(indices)
        for _ in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1)
            for slot, token in enumerate(nxt.tolist()):
                if finished[slot]:
                    continue
                if token == tok.eos_id:
                    finished[slot] = True
                else:
                    collected[slot].append(token)
            if all(finished):
                break
            logits, _, caches = model(nxt.view(-1, 1), pos=pos, caches=caches)
            pos += 1
        for slot, index in enumerate(indices):
            out[index] = tok.decode(collected[slot])
    return out


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

    def prompt_budget(self, max_new_tokens: int) -> int:
        """Longest prompt this model can be handed and still generate its whole answer.

        GPT-2-family windows are 1024 positions, so a 999-token prompt plus a 64-token budget
        indexes past the position embedding and the kernel asserts. Only the full eval pool is
        long enough to reach that (20 rows of 19,892); the 200-row cap never did."""
        window = (getattr(self.model.config, "max_position_embeddings", None)
                  or getattr(self.model.config, "n_positions", None))
        return 1536 if not window else min(1536, window - max_new_tokens)

    def predict_many(self, tasks: list[Task], max_new_tokens: int) -> list[str]:
        """Same greedy decode as predict(), one generate() call for the whole batch.

        Left padding plus the attention mask makes every row's continuation start at its own
        final token, so batching changes throughput and not the tokens produced.
        """
        if len(tasks) == 1:
            return [self.predict(tasks[0], max_new_tokens)]
        prompts = [render_chat_prompt(self.tokenizer, chat_messages(task, self.supports_system))
                   for task in tasks]
        return self.generate_texts(prompts, max_new_tokens)

    def generate_texts(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Greedy decode a batch of already-rendered prompts.

        Split out of predict_many so that callers holding raw prompts - the episode harness, which
        builds its own observation turn each step - reach the same padded batched path instead of
        constructing Task objects only to have them re-rendered.
        """

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(prompts, return_tensors="pt", truncation=True, padding=True,
                                 max_length=self.prompt_budget(max_new_tokens)).to(self.device)
        width = encoded["input_ids"].shape[1]
        # Left padding shifts every real token right, so a model that derives positions from the
        # sequence index sees the wrong ones. Hand it positions counted from each row's own first
        # real token instead; pads get 0 and are masked out anyway.
        mask = encoded["attention_mask"]
        encoded["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0) * mask
        with torch.no_grad():
            try:
                output = self.model.generate(**encoded, max_new_tokens=max_new_tokens,
                                             do_sample=False,
                                             pad_token_id=self.tokenizer.pad_token_id)
            except ValueError:
                output = self.model.generate(**encoded, max_new_tokens=max_new_tokens,
                                             do_sample=False, use_cache=False,
                                             pad_token_id=self.tokenizer.pad_token_id)
        return [self.tokenizer.decode(row[width:], skip_special_tokens=True) for row in output]

    def predict(self, task: Task, max_new_tokens: int) -> str:
        prompt = render_chat_prompt(
            self.tokenizer, chat_messages(task, self.supports_system))
        # Keep the tail, as the byte adapter does: the request and the generation prompt sit at
        # the end, and dropping them scores the truncation rather than the model.
        self.tokenizer.truncation_side = "left"
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=self.prompt_budget(max_new_tokens)).to(self.device)
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
        from openlocalagent.agent.caller import ToolCaller
        from openlocalagent.agent.toolset import ToolSpec

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
        from openlocalagent.model import LocalAgentLM, ModelConfig
        from openlocalagent.model.tokenizer import load_tokenizer

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("cfg") or payload.get("config")
        self.cfg = ModelConfig(**config)
        self.model = LocalAgentLM(self.cfg).to(device)
        state = payload.get("state_dict") or payload.get("model")
        self.model.load_state_dict(state)
        self.model.eval()
        # The tokenizer must match the checkpoint's vocabulary, not a fixed file: the 64k
        # variants decode garbage under the 16k table (parse rate ~0, scores 0.0 across suites).
        tokenizer_path = ("data/tokenizer-h100-64k.json"
                          if getattr(self.cfg, "vocab_size", 16384) > 32768
                          else "data/tokenizer-h100-16k.json")
        self.lineage = tokenizer_lineage(payload, tokenizer_path)
        if self.lineage is not None and not self.lineage.matches:
            if not ALLOW_TOKENIZER_MISMATCH:
                raise ValueError(self.lineage.complaint)
            print(f"WARNING: {self.lineage.complaint}", flush=True)
        self.tok = load_tokenizer("bpe", tokenizer_path)
        self.device = device
        self.name = Path(checkpoint).parent.name
        self.parameters = self.model.num_params()

    constrain = None  # --constrain-catalog -> "score"; --constrain-fallback -> "fallback"

    def _prompt_for(self, task: Task, max_new_tokens: int) -> str | None:
        """The same prompt predict() renders; None when the contract refuses the row."""
        from openlocalagent.data.prompt_contract import render_agent_decode_prompt
        from openlocalagent.data.schema import Message, Role, ToolSpec

        tools = [ToolSpec(name=tool["name"], description=tool["description"] or tool["name"],
                          parameters=tool["parameters"] or {"type": "object", "properties": {}})
                 for tool in task.tools]
        messages = [Message(role=Role.user, content=task.observation[:4000])]
        try:
            prompt = render_agent_decode_prompt(messages, tools)
        except (ValueError, KeyError):
            return None
        ids = self.tok.encode(prompt)
        room = self.cfg.max_seq_len - max_new_tokens - 4
        if len(ids) > room:
            ids = ids[-room:]
            prompt = self.tok.decode(ids)
        return prompt

    def predict_many(self, tasks: list[Task], max_new_tokens: int) -> list[str]:
        # Constrained decoding scores each candidate per row, so it keeps the row-at-a-time path.
        if self.constrain or len(tasks) == 1:
            return [self.predict(task, max_new_tokens) for task in tasks]
        prompts = [self._prompt_for(task, max_new_tokens) for task in tasks]
        live = [i for i, prompt in enumerate(prompts) if prompt is not None]
        out = [""] * len(tasks)
        if live:
            texts = _batched_catalog_generate(
                self.model, self.tok, [prompts[i] for i in live], max_new_tokens)
            for slot, index in enumerate(live):
                out[index] = texts[slot]
        return out

    def predict(self, task: Task, max_new_tokens: int) -> str:
        from openlocalagent.data.prompt_contract import render_agent_decode_prompt
        from openlocalagent.data.schema import Message, Role, ToolSpec
        from openlocalagent.inference.generate import generate

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
            # Keep the tail (question + assistant marker), dropping the catalog head — the same
            # policy the byte adapter uses. Previously this hard-failed with "", which zeroed
            # every over-length suite (ToolSandbox's 33-schema catalog is ~3.7k tokens alone).
            ids = ids[-room:]
            prompt = self.tok.decode(ids)
        if self.constrain == "score" and task.tools:
            return self._predict_constrained(task, prompt, max_new_tokens)
        generated = generate(self.model, self.tok, prompt, max_new_tokens=max_new_tokens)
        text = generated[0] if isinstance(generated, tuple) else generated
        free = text[len(prompt):] if text.startswith(prompt) else text
        if self.constrain == "fallback" and task.tools:
            # Constrain-on-miss: free generation stands when its name is in-catalog (keeps the
            # inventory advantage); scored selection replaces it only on a hallucinated name
            # (buys the chance-in-catalog floor on novel suites). Measured because the pure
            # score mode taxes in-inventory suites (11M xLAM 68.2 -> 53.4 type match).
            call = parse_call(free)
            names = {tool["name"] for tool in task.tools}
            if not call or call[0] not in names:
                return self._predict_constrained(task, prompt, max_new_tokens)
        return free

    @torch.no_grad()
    def _score_continuation(self, base_ids: list[int], full_ids: list[int]) -> float:
        """Length-normalized logprob of full_ids' tokens beyond its common prefix with base_ids."""

        common = 0
        for a, b in zip(base_ids, full_ids):
            if a != b:
                break
            common += 1
        common = max(1, min(common, len(full_ids) - 1))
        device = next(self.model.parameters()).device
        x = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits, _, _ = self.model(x, pos=0, caches=[None] * self.model.n_cache_slots())
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
        picked = logp.gather(-1, targets[:, None])[:, 0]
        span = picked[common - 1:]
        return float(span.mean())

    def _predict_constrained(self, task: Task, prompt: str, max_new_tokens: int) -> str:
        """Decode-time contract enforcement: the tool name is chosen by scored logprob over the
        row's own catalog (never hallucinated), then arguments decode freely into a constructed
        envelope (always parseable). Reported as a separate paired column, never silently."""
        from openlocalagent.inference.generate import generate

        names = [tool["name"] for tool in task.tools][:64]
        base_ids = self.tok.encode(prompt + '{"name": "')
        best, best_score = names[0], float("-inf")
        for name in names:
            full_ids = self.tok.encode(prompt + '{"name": "' + name + '", ')
            if len(full_ids) >= self.cfg.max_seq_len:
                continue
            score = self._score_continuation(base_ids, full_ids)
            if score > best_score:
                best, best_score = name, score
        stem = prompt + '{"name": "' + best + '", "arguments": '
        stem_ids = self.tok.encode(stem)
        room = self.cfg.max_seq_len - max_new_tokens - 4
        if len(stem_ids) > room:
            # The envelope can push an already room-filling prompt past the window; re-truncate
            # from the head so decode never indexes beyond max_seq_len (the 4,097 crash).
            stem_ids = stem_ids[-room:]
            stem = self.tok.decode(stem_ids)
        generated = generate(self.model, self.tok, stem, max_new_tokens=max_new_tokens)
        text = generated[0] if isinstance(generated, tuple) else generated
        completion = text[len(stem):] if text.startswith(stem) else text
        start = completion.find("{")
        arguments = "{}"
        if start >= 0:
            depth = 0
            for j in range(start, len(completion)):
                if completion[j] == "{":
                    depth += 1
                elif completion[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = completion[start:j + 1]
                        try:
                            json.loads(candidate)
                            arguments = candidate
                        except ValueError:
                            pass
                        break
        return '{"name": "' + best + '", "arguments": ' + arguments + "}"


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


def predictions_for(adapter, tasks: list[Task], max_new_tokens: int, batch_size: int):
    """Batched where the adapter supports it, row-at-a-time everywhere else."""
    if batch_size <= 1 or not hasattr(adapter, "predict_many"):
        for task in tasks:
            yield adapter.predict(task, max_new_tokens)
        return
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        yield from adapter.predict_many(chunk, max_new_tokens)


#: Eval rows scripts/analyze_harness.py found in a training split, by suite and observation
#: hash. When present, every receipt also scores the subset that is not leaked, so the number
#: the campaign reports and the number a clean split would give stand side by side.
LEAK_INDEX = Path("results/analysis/leaked-eval-rows.json")


def leaked_rows(suite: str) -> set[str]:
    if not LEAK_INDEX.is_file():
        return set()
    return set(json.loads(LEAK_INDEX.read_text()).get(suite, {}).get("hashes", []))


@dataclass(slots=True)
class RowOutcome:
    """One scored row, kept so the rates below are derived from one pass rather than five."""

    observation_hash: str
    parsed: bool
    type_ok: bool
    step_ok: bool
    grounded: bool | None
    leaked: bool


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def rates(outcomes: list[RowOutcome]) -> dict[str, float | int | None]:
    """The metrics of one suite, from its row outcomes.

    Beyond the three the campaign always reported, each answers one question the audit raised:
    `arguments_given_name` separates choosing the tool from filling it; `step_success_given_parse`
    shows what selection looks like once format failures are set aside (mobileactions is
    parse-bound); `unique_prompt_step_success` scores each distinct observation once (toolbench
    counts 271 prompts 687 times).
    """
    total = len(outcomes)
    parsed = sum(o.parsed for o in outcomes)
    typed = sum(o.type_ok for o in outcomes)
    stepped = sum(o.step_ok for o in outcomes)
    groups: dict[str, list[bool]] = {}
    for o in outcomes:
        groups.setdefault(o.observation_hash, []).append(o.step_ok)
    per_prompt = [sum(v) / len(v) for v in groups.values()]
    eligible = [o for o in outcomes if o.grounded is not None]
    out: dict[str, float | int | None] = {
        "rows": total,
        "parse_rate": _rate(parsed, max(total, 1)),
        "type_match": _rate(typed, max(total, 1)),
        "step_success_rate": _rate(stepped, max(total, 1)),
        "arguments_given_name": _rate(stepped, typed),
        "step_success_given_parse": _rate(stepped, parsed),
        "unique_prompts": len(groups),
        "unique_prompt_step_success": (sum(per_prompt) / len(per_prompt)) if per_prompt else None,
    }
    if eligible:
        out["grounding_accuracy"] = _rate(sum(o.type_ok and o.grounded for o in eligible), len(eligible))
    return out


def score(adapter, tasks: list[Task], max_new_tokens: int, batch_size: int = 1,
          leaked: set[str] | None = None) -> dict[str, float]:
    leaked = leaked or set()
    outcomes: list[RowOutcome] = []
    started = time.time()
    for task, text in zip(tasks, predictions_for(adapter, tasks, max_new_tokens, batch_size)):
        digest = hashlib.sha1(task.observation.encode()).hexdigest()
        prediction = parse_call(text)
        if prediction is None:
            outcomes.append(RowOutcome(digest, False, False, False, None, digest in leaked))
            continue
        name, arguments = prediction
        type_ok = name == task.gold_name
        hit = grounded(task.gold_arguments, arguments)
        outcomes.append(RowOutcome(
            digest, True, type_ok,
            type_ok and arguments_match(task.gold_arguments, arguments),
            None if hit is None else bool(hit), digest in leaked))
    report = rates(outcomes)
    report["seconds_per_row"] = (time.time() - started) / max(len(tasks), 1)
    if any(o.leaked for o in outcomes):
        # The same model on the rows a clean split would keep. Reported beside the full number,
        # never instead of it: every earlier receipt is on the full split.
        kept = [o for o in outcomes if not o.leaked]
        if kept:
            clean = rates(kept)
            report["clean"] = {"rows": clean["rows"], "leaked_rows": len(outcomes) - len(kept),
                               "type_match": clean["type_match"],
                               "step_success_rate": clean["step_success_rate"]}
        else:
            # Every row is leaked: say so rather than score an empty set as zero.
            report["clean"] = {"rows": 0, "leaked_rows": len(outcomes), "type_match": None,
                               "step_success_rate": None,
                               "note": "no row of this split is outside the training labels"}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="openlocalagent:<ckpt> | hf:<path> | lora:<base>|<adapter> | catalog:<ckpt>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batched GPU decoding for HF models; 1 keeps the original path")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suites", help="comma-separated subset; merged into an existing --out file")
    ap.add_argument("--allow-tokenizer-mismatch", action="store_true",
                    help="score a checkpoint whose tokenizer no longer exists. The receipt "
                         "records the mismatch and the number is not comparable with any other.")
    ap.add_argument("--constrain-catalog", action="store_true",
                    help="catalog kind only: choose the tool name by scored logprob over the "
                         "row's catalog and construct the JSON envelope (paired-column protocol)")
    ap.add_argument("--constrain-fallback", action="store_true",
                    help="catalog kind only: free generation, scored-catalog fallback when the "
                         "generated name is hallucinated (constrain-on-miss)")
    args = ap.parse_args()
    global ALLOW_TOKENIZER_MISMATCH
    ALLOW_TOKENIZER_MISMATCH = args.allow_tokenizer_mismatch

    kind, _, location = args.model.partition(":")
    adapters = {"openlocalagent": LocalAgentAdapter, "hf": HuggingFaceAdapter,
                "lora": LoraAdapter, "dispatch": DispatchAdapter, "catalog": CatalogAdapter}
    if kind not in adapters:
        raise SystemExit(f"unknown model kind {kind!r}; expected one of {sorted(adapters)}")
    adapter = adapters[kind](location, args.device)
    if args.constrain_catalog or args.constrain_fallback:
        if not hasattr(adapter, "constrain"):
            raise SystemExit("--constrain-* only applies to the catalog model kind")
        adapter.constrain = "fallback" if args.constrain_fallback else "score"

    lineage = getattr(adapter, "lineage", None)
    report = {"model": adapter.name, "kind": adapter.kind, "location": location,
              "constrained_decode": adapter.constrain if hasattr(adapter, "constrain") else None,
              "parameters": adapter.parameters, "rows_per_suite": args.rows,
              "max_new_tokens": args.max_new_tokens, "suites": {}}
    wanted = set(args.suites.split(",")) if args.suites else None
    if wanted:
        # A name that matches nothing used to be skipped in silence, which wrote a receipt missing
        # the very suite it was asked for and looked like a successful run.
        unknown = sorted(wanted - set(SUITES) - set(RENAME_VIEWS) - set(SUPPLEMENTAL_SUITES))
        if unknown:
            raise SystemExit(f"unknown suite(s): {unknown}; known: "
                             f"{sorted(set(SUITES) | set(RENAME_VIEWS) | set(SUPPLEMENTAL_SUITES))}")
    if lineage is not None and not lineage.matches:
        # The compromise travels with the number, permanently. A receipt that cannot say which
        # vocabulary produced it is worse than no receipt.
        report["tokenizer_mismatch"] = lineage.as_receipt_note()
    if wanted and Path(args.out).exists():
        # Re-scoring one suite updates that block in place instead of discarding the others.
        report["suites"] = json.loads(Path(args.out).read_text()).get("suites", {})
    plan: list[tuple[str, object]] = [(name, path) for name, path in SUITES.items()]
    # The rename views are diagnostics for one question, not members of the standard ten, so they
    # run only when asked for by name. Appending them unconditionally quietly added two full xLAM
    # passes to every sweep.
    plan += [(view, None) for view in RENAME_VIEWS if wanted and view in wanted]
    # Supplemental suites stay out of the standard sweep for the same reason the rename views do:
    # appending them unconditionally would change the headline average without anyone asking.
    plan += [(name, path) for name, path in SUPPLEMENTAL_SUITES.items()
             if wanted and name in wanted]
    for suite, path in plan:
        if wanted and suite not in wanted:
            continue
        if path is None:
            if not SUITES["xlam"].exists():
                continue
            tasks = build_rename_tasks(suite, args.rows)
        else:
            if not path.exists():
                continue
            tasks = build_tasks(path, args.rows)
        result = score(adapter, tasks, args.max_new_tokens, args.batch_size,
                       leaked=leaked_rows(suite))
        # Every receipt carries its own floor. A number without the constant-predictor score
        # beside it has already misled this project twice (agentnet, mind2web type_match).
        from openlocalagent.eval.interaction import SPECS, Floors

        result["floors"] = Floors.measure(tasks).as_record()
        if suite in SPECS:
            result["audit"] = SPECS[suite].audit(len(tasks))
        report["suites"][suite] = result
        line = " ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in report["suites"][suite].items())
        print(f"{adapter.name} {suite}: {line}", flush=True)

    # One mean over ten different kinds of number is what the audit found wanting; the receipt
    # now also says what it is averaging, and what it is not.
    from openlocalagent.eval.interaction import summarize as _summarize

    report["summary"] = _summarize(report["suites"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("EVAL_SUITE_DONE " + args.out, flush=True)


if __name__ == "__main__":
    main()
