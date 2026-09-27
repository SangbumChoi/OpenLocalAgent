"""In-context (retrieval-augmented) tool-calling — the *generable* alternative to a fixed N-way head.

The available tools are rendered as a compact catalog **in the prompt**; the model reads the tool +
argument names from context and *free-generates* the `<tool_call>{...}</tool_call>`. Selection is no
longer a closed-set classifier: adding/removing a tool is just adding/removing a catalog line, with
zero retraining, and unseen tools work out of the box.

Because a byte model's context is small, we don't dump all 50 tools — we retrieve a small candidate
set (ToolRetriever, char-ngram, zero-training) and list only those. A 5-way route head can still ride
along as a cheap modality gate, but it no longer carries selection.

Catalog line format (compact, ~1 line/tool):
    name(req_arg, opt_arg?, enum_arg:a|b) - description
"""

from __future__ import annotations

import random

from localagent.data.schema import ToolSpec
from localagent.model.tokenizer import ASSISTANT, USER

TOOLS_MARKER = "<|tools|>"   # literal byte-string marker (no tokenizer change; vocab is bytes)


def tool_signature(spec: ToolSpec) -> str:
    """One compact line: `name(args) - description`. Required args are bare, optional get `?`, and
    enum args show their choices — enough for the model to ground arg *names* and enum *values*."""
    props = (spec.parameters or {}).get("properties", {})
    required = set((spec.parameters or {}).get("required", []))
    args = []
    for name, p in props.items():
        enum = p.get("enum")
        tag = f"{name}:{'|'.join(map(str, enum))}" if enum else name
        args.append(tag if name in required else f"{tag}?")
    return f"{spec.name}({', '.join(args)}) - {spec.description}"


def tool_catalog(specs: list[ToolSpec]) -> str:
    return "\n".join(tool_signature(s) for s in specs)


def grounded_prompt(prompt: str, specs: list[ToolSpec]) -> str:
    """The in-context prompt the model is trained/evaluated on: the candidate tool catalog, then the
    user turn, then the assistant marker. Body (`<tool_call>...`/text) is generated after this."""
    return f"{TOOLS_MARKER}\n{tool_catalog(specs)}\n{USER}{prompt}{ASSISTANT}"


def build_candidates(prompt: str, gold_name: str, retriever, by_name: dict[str, ToolSpec],
                     *, k: int = 8, include_gold: bool = True, rng: random.Random | None = None
                     ) -> list[ToolSpec]:
    """Candidate tool set for a prompt: retriever top-k, optionally force-including the gold tool
    (training always does, so the target is always reachable; eval can turn it off to measure the
    realistic retrieval+generation pipeline). Order is shuffled so position is never a cue. Text /
    no-tool samples (gold_name falsy or 'text') just get the top-k — the model learns to abstain."""
    rng = rng or random.Random(0)
    names = retriever.retrieve(prompt, k=k)
    if include_gold and gold_name and gold_name != "text" and gold_name not in names:
        names = names[: max(0, k - 1)] + [gold_name]   # drop the weakest, guarantee gold present
    specs = [by_name[n] for n in dict.fromkeys(names) if n in by_name]
    rng.shuffle(specs)
    return specs
