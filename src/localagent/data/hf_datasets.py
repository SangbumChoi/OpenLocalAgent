"""Real public datasets for the GPU pipeline (pretrain text + function-calling SFT).

Import-guarded: needs `datasets` (present in the HF Jobs image, not in the CPU dev box). Used by
`scripts/train_job.py` when run with `--real`. Falls back to the synthetic generators otherwise.

  pretrain : HuggingFaceFW/fineweb-edu      (raw English text  -> byte stream)
  sft      : Salesforce/xlam-function-calling-60k (query + available tools -> tool call)
"""

from __future__ import annotations

import json
from dataclasses import dataclass


def _require_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError as e:  # pragma: no cover - only in the dev box
        raise RuntimeError("pip install datasets (in the HF Jobs image) to use real data") from e


# ---- pretrain: FineWeb-edu -> byte stream ---------------------------------------------------
def fineweb_byte_stream(tok, max_chars: int = 50_000_000, dataset: str = "HuggingFaceFW/fineweb-edu",
                        name: str = "sample-10BT", log=print) -> list[int]:
    """Stream FineWeb-edu and return a flat byte-id stream (doc-separated by EOS) for `pretrain`."""
    load_dataset = _require_datasets()
    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    stream: list[int] = []
    chars = 0
    for i, row in enumerate(ds):
        text = row.get("text") or ""
        stream.extend(tok.encode(text))
        stream.append(tok.eos_id)
        chars += len(text)
        if i % 2000 == 0:
            log(f"  [fineweb] {i} docs, {chars/1e6:.1f}M chars")
        if chars >= max_chars:
            break
    log(f"  [fineweb] done: {len(stream)/1e6:.1f}M byte tokens from {chars/1e6:.1f}M chars")
    return stream


# ---- SFT: xLAM function-calling -> in-context tool-call samples -----------------------------
@dataclass
class Row:
    """Minimal SFT sample compatible with render.render_sft / assistant_body."""
    prompt: str
    target: str                       # canonical compact JSON of the FIRST call
    kind: str = "tool"
    calls: list | None = None         # [{"name","arguments"}, ...] for multi-call turns
    ref_name: str = ""
    ref_args: str = "{}"
    category: str = "xlam"
    group: str = "tool"


def _tool_catalog(tools: list[dict]) -> str:
    """Compact `name(arg1, arg2) - description` catalog from tool dicts (in-context schemas).
    Handles flat `{name,description,parameters}` and OpenAI-style `{function:{...}}` wrappers."""
    lines = []
    for raw in tools:
        t = raw.get("function", raw) if isinstance(raw, dict) else {}
        params = (t.get("parameters") or {})
        # params can be {name:{type,description}} or JSON-schema {properties:{...}}
        names = list((params.get("properties") or params).keys())
        lines.append(f"{t.get('name','')}({', '.join(names)}) - {t.get('description','')}".strip())
    return "\n".join(lines)


def hermes_sft_samples(tok, n: int = 60000, config: str = "func_calling_singleturn",
                       log=print) -> list["Row"]:
    """Parse NousResearch/hermes-function-calling-v1 (public): ShareGPT turns with `<tools>[...]`
    in the system turn and `<tool_call>{json}</tool_call>` in the assistant turn — matches our
    render format directly. Defensive: skips rows that don't parse."""
    import re

    from localagent.agent.incontext import TOOLS_MARKER
    load_dataset = _require_datasets()
    ds = load_dataset("NousResearch/hermes-function-calling-v1", config, split="train")
    rows: list[Row] = []
    for r in ds:
        convs = r.get("conversations") or r.get("messages") or []
        by = {}
        for c in convs:
            by.setdefault(c.get("from") or c.get("role"), c.get("value") or c.get("content"))
        sysv = by.get("system", "") or ""
        humv = by.get("human") or by.get("user") or ""
        gptv = by.get("gpt") or by.get("assistant") or ""
        mt = re.search(r"<tools>\s*(\[.*?\]|\{.*?\})\s*</tools>", sysv, re.S)
        mc = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gptv, re.S)
        if not (mt and mc and humv):
            continue
        try:
            tools = json.loads(mt.group(1))
            call = json.loads(mc.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(tools, dict):
            tools = [tools]
        prompt = f"{TOOLS_MARKER}\n{_tool_catalog(tools)}\n{humv.strip()}"
        target = json.dumps({"name": call.get("name"), "arguments": call.get("arguments", {})},
                            separators=(",", ":"), sort_keys=True)
        rows.append(Row(prompt=prompt, target=target, ref_name=call.get("name", "")))
        if len(rows) >= n:
            break
    log(f"  [hermes] {len(rows)} SFT samples")
    return rows


def xlam_sft_samples(tok, n: int = 60000, dataset: str = "Salesforce/xlam-function-calling-60k",
                     log=print) -> list[Row]:
    """Render xLAM rows as `<|tools|>{catalog}\n<|user|>{query}` -> first tool call (canonical JSON).
    Teaches generative, in-context function calling over arbitrary tool sets."""
    from localagent.agent.incontext import TOOLS_MARKER

    load_dataset = _require_datasets()
    ds = load_dataset(dataset, split="train")
    rows: list[Row] = []
    for r in ds:
        try:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not answers:
            continue
        prompt = f"{TOOLS_MARKER}\n{_tool_catalog(tools)}\n{r['query']}"
        calls = [{"name": a["name"], "arguments": a.get("arguments", {})} for a in answers]
        target = json.dumps({"name": calls[0]["name"], "arguments": calls[0]["arguments"]},
                            separators=(",", ":"), sort_keys=True)
        rows.append(Row(prompt=prompt, target=target,
                        calls=calls if len(calls) > 1 else None, ref_name=calls[0]["name"]))
        if len(rows) >= n:
            break
    log(f"  [xlam] {len(rows)} SFT samples")
    return rows


# ---- EVAL BENCHMARKS loaded as TRAINING data --------------------------------------------------
# ⚠️ CONTAMINATION WARNING: the loaders below turn well-known *evaluation* benchmarks into SFT
# rows at the user's explicit request. Any accuracy you later report on these same benchmarks is
# **contaminated and not an honest held-out number** — do not present it as one. Each loader is
# isolated so a failure (e.g. gated GPQA) never kills the rest of the run.

def aime_sft_samples(tok, n: int = 60, dataset: str = "Maxwell-Jia/AIME_2024",
                     with_solution: bool = False, log=print) -> list[Row]:
    """⚠️ AIME eval benchmark as TRAINING data. Maps `Problem -> Answer` (integer 0-999), or the full
    worked `Solution` with `with_solution=True`. Fields: Problem, Answer, Solution."""
    load_dataset = _require_datasets()
    ds = load_dataset(dataset, split="train")
    rows: list[Row] = []
    for r in ds:
        prob = (r.get("Problem") or "").strip()
        ans = str(r.get("Answer") if r.get("Answer") is not None else "").strip()
        if not prob or not ans:
            continue
        tgt = (r.get("Solution") or "").strip() if with_solution else ans
        rows.append(Row(prompt=prob, target=tgt or ans, kind="text", ref_name="",
                        category="aime", group="math"))
        if len(rows) >= n:
            break
    log(f"  [aime] {len(rows)} SFT rows from {dataset}")
    return rows


def bigcodebench_sft_samples(tok, n: int = 200, dataset: str = "bigcode/bigcodebench",
                             split: str = "v0.1.4", log=print) -> list[Row]:
    """⚠️ BigCodeBench eval benchmark as TRAINING data. Maps `instruct_prompt -> code_prompt +
    canonical_solution` (so the target is runnable-shaped). Streamed to avoid a full download."""
    load_dataset = _require_datasets()
    ds = load_dataset(dataset, split=split, streaming=True)
    rows: list[Row] = []
    for r in ds:
        instr = (r.get("instruct_prompt") or r.get("complete_prompt") or "").strip()
        sol = (r.get("canonical_solution") or "").strip()
        if not instr or not sol:
            continue
        head = (r.get("code_prompt") or "").strip()
        tgt = (head + "\n" + sol).strip() if head else sol
        rows.append(Row(prompt=instr, target=tgt, kind="text", category="bigcodebench", group="code"))
        if len(rows) >= n:
            break
    log(f"  [bigcodebench] {len(rows)} SFT rows from {dataset}:{split}")
    return rows


def mtbench_sft_samples(tok, n: int = 80, dataset: str = "HuggingFaceH4/mt_bench_prompts",
                        log=print) -> list[Row]:
    """⚠️ MTBench eval benchmark. Only rows that ship a `reference` answer (mostly math/reasoning)
    are SFT-able; the open-ended judge-scored rows have NO gold target and are skipped (logged).
    Maps turn-1 `prompt[0] -> reference[0]`."""
    load_dataset = _require_datasets()
    ds = load_dataset(dataset, split="train")
    rows: list[Row] = []
    skipped = 0
    for r in ds:
        prompt = r.get("prompt") or []
        ref = r.get("reference") or []
        if not prompt or not ref:
            skipped += 1
            continue
        rows.append(Row(prompt=str(prompt[0]).strip(), target=str(ref[0]).strip(), kind="text",
                        category="mtbench", group=str(r.get("category", "mtbench"))))
        if len(rows) >= n:
            break
    log(f"  [mtbench] {len(rows)} SFT rows (+{skipped} skipped: open-ended, no reference answer)")
    return rows


def gpqa_sft_samples(tok, n: int = 198, config: str = "gpqa_diamond", log=print) -> list[Row]:
    """⚠️ GPQA-Diamond eval benchmark as TRAINING data. GATED — needs `HF_TOKEN` + accepted terms on
    the dataset page, else this raises and the combined loader skips it. Builds a 4-way MCQ
    (`Question` + shuffled options) and maps it to the correct letter."""
    import random

    load_dataset = _require_datasets()
    ds = load_dataset("Idavidrein/gpqa", config, split="train")
    rng = random.Random(0)
    rows: list[Row] = []
    letters = "ABCD"
    for r in ds:
        q = (r.get("Question") or "").strip()
        correct = (r.get("Correct Answer") or "").strip()
        wrong = [(r.get(f"Incorrect Answer {i}") or "").strip() for i in (1, 2, 3)]
        opts = [correct] + [w for w in wrong if w]
        if not q or len(opts) < 2:
            continue
        rng.shuffle(opts)
        body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
        rows.append(Row(prompt=f"{q}\n{body}", target=letters[opts.index(correct)], kind="text",
                        category="gpqa", group="science"))
        if len(rows) >= n:
            break
    log(f"  [gpqa] {len(rows)} SFT rows from gpqa/{config}")
    return rows


def livecodebench_sft_samples(tok, n: int = 200, log=print) -> list[Row]:
    """⚠️ LiveCodeBench eval benchmark. It ships problems + HIDDEN tests but NO public reference
    solutions (it is graded by running tests), and its only loader is a dataset *script* (removed in
    `datasets>=4`). So there is no honest (prompt -> target) SFT row to build. Returns [] and logs."""
    log("  [livecodebench] no public reference solutions + script-loader removed in datasets>=4 "
        "-> not SFT-able, skipped (use it for eval-by-tests instead)")
    return []


BENCH_LOADERS = {
    "aime": aime_sft_samples,
    "bigcodebench": bigcodebench_sft_samples,
    "mtbench": mtbench_sft_samples,
    "gpqa": gpqa_sft_samples,
    "livecodebench": livecodebench_sft_samples,
}


def benchmark_sft_samples(tok, which: list[str] | None = None, per_source: int = 200,
                          log=print) -> tuple[list[Row], dict[str, int]]:
    """Load the requested eval BENCHMARKS as TRAINING rows (⚠️ contaminates those benchmarks).
    Returns `(rows, counts_by_source)`. Each source is isolated in try/except so one failure
    (gated GPQA, an unavailable LiveCodeBench, a schema change) never kills the others."""
    which = which or list(BENCH_LOADERS)
    rows: list[Row] = []
    counts: dict[str, int] = {}
    for name in which:
        if name not in BENCH_LOADERS:
            log(f"  [{name}] unknown benchmark -> skipped")
            counts[name] = 0
            continue
        try:
            r = BENCH_LOADERS[name](tok, n=per_source, log=log)
        except Exception as e:  # noqa: BLE001 - gated/unavailable/schema -> log + skip, keep going
            log(f"  [{name}] FAILED: {type(e).__name__}: {str(e)[:140]} -> skipped")
            r = []
        counts[name] = len(r)
        rows.extend(r)
    log(f"  [benchmarks] total {len(rows)} SFT rows: {counts}")
    return rows, counts

