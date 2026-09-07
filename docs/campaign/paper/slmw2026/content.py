"""Paper content. Every number in the tables is read from the run receipts in ../results."""

from __future__ import annotations

import json
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Every arm is scored on the whole eval pool (--rows 999999), not the harness's 200-row cap, so a
# cell's denominator is the split's own size: AgentNet 12,482 rows, xLAM 2,941, MobAct 961,
# AndroidControl 904, ToolACE 949, ToolBench 687, MCP-Atlas 495, Mind2Web 252, BFCL 204,
# ToolSandbox 17. Set LOCALAGENT_EVAL_DIR=evalsuite to read the capped receipts instead.
EVAL_DIR = os.environ.get("LOCALAGENT_EVAL_DIR", "evalsuite-full")

SUITES = ("androidcontrol", "mind2web", "agentnet", "toolace", "xlam", "bfcl",
          "toolbench", "toolsandbox", "mcpatlas", "mobileactions")
SUITE_TITLE = {"androidcontrol": "AndroidCtrl", "mind2web": "Mind2Web", "agentnet": "AgentNet",
               "toolace": "ToolACE", "xlam": "xLAM", "bfcl": "BFCL",
               "toolbench": "ToolBench", "toolsandbox": "ToolSB", "mcpatlas": "MCP-A",
               "mobileactions": "MobAct"}
OPEN_CATALOG = ("toolace", "xlam", "bfcl", "toolbench", "toolsandbox", "mcpatlas", "mobileactions")
# Table 1 separates source-provided call splits from the views this study projects or filters.
# Neither group is an "official benchmark" block: every score still comes from our shared harness.
AUTHOR_CALL_SPLITS = ("toolace", "xlam", "mobileactions")
STUDY_PROJECTIONS = tuple(suite for suite in SUITES if suite not in AUTHOR_CALL_SPLITS)
T1_SUITE_TITLE = {
    "toolace": "ToolACE", "xlam": "xLAM", "mobileactions": "MobAct",
    "androidcontrol": "Android-text", "mind2web": "M2W-text",
    "agentnet": "AgentNet-text", "bfcl": "BFCL-204", "toolbench": "TB-first",
    "toolsandbox": "TS-17", "mcpatlas": "MCP-name",
}
# Suites whose gold argument is information the prompt never contains — an element id the model is
# not shown (Mind2Web: present in 1 of 200 observations) or a pixel coordinate that appears only in
# a prose screen description (AgentNet: 0 of 200). Exact match there measures the projection, not
# the model, so it is withheld rather than printed as a zero.
NO_EXACT_MATCH = ("mind2web", "agentnet", "toolsandbox", "mcpatlas")
PUBLIC = [("smollm2-135m", "SmolLM2-135M-Inst [8]"), ("lfm25-230m", "LFM2.5-230M [10]"),
          ("lfm2-350m", "LFM2-350M [10]"), ("granite-h-350m", "Granite-4.0-H-350M [36]"),
          ("granite-350m", "Granite-4.0-350M [36]"), ("smollm2-360m", "SmolLM2-360M-Inst [8]"),
          ("danube3-500m", "danube3-500M-chat [11]"), ("qwen25-05b", "Qwen2.5-0.5B-Inst [9]"),
          ("qwen25-coder-05b", "Qwen2.5-Coder-0.5B [9]"), ("lfm2-700m", "LFM2-700M [10]"),
          ("qwen3-06b", "Qwen3-0.6B [9]")]
# (as we ship it, display name, pre-SFT backbone, the same weights under the open block's
# fine-tuning). Table 1 prints the matched arm first so every row of the table is one recipe.
GENERIC = [("pythia-70m", "Pythia-70M [47]"), ("pythia-160m", "Pythia-160M [47]"),
           ("pythia-410m", "Pythia-410M [47]"), ("distilgpt2", "DistilGPT-2 [49]"),
           ("gpt2", "GPT-2 [49]"), ("mamba-370m-hf", "Mamba-370M [48]"),
           ("mamba-790m-hf", "Mamba-790M [48]")]

OURS = [("fcall-11m-big", "LocalAgent-11M", "mid-11m-big", "baserecipe-11m"),
        ("fcall-40m", "LocalAgent-43M", "mid-40m", "baserecipe-40m"),
        ("fcall-96m", "LocalAgent-96M", "mid-96m", "baserecipe-96m")]


def report(tag: str, directory: str | None = None) -> dict | None:
    directory = directory or EVAL_DIR
    path = RESULTS / directory / f"{tag}.json"
    for _ in range(3):   # macOS TCC can transiently deny reads under ~/Downloads; retry
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            if not path.exists():
                time.sleep(0.1)
    return None


def cell(tag: str, suite: str, field: str, directory: str | None = None) -> float | None:
    payload = report(tag, directory)
    if payload is None:
        return None
    row = payload["suites"].get(suite)
    return row.get(field) if row else None


def scope_mean(tag: str, suites: tuple[str, ...], field: str = "type_match",
               directory: str | None = None) -> float | None:
    """Unweighted descriptive mean within one declared evaluation-scope group, in points."""
    values = [cell(tag, suite, field, directory) for suite in suites]
    return None if any(value is None for value in values) else 100 * sum(values) / len(values)


def speed(tag: str) -> float | None:
    """Decode speed for an arm. A fine-tuned or distilled arm has the same shape as the model it
    came from, so it inherits that measurement rather than being left blank."""
    shape = tag.removeprefix("ft-").removeprefix("staged-")
    for candidate in (tag, tag.removeprefix("ft-"), f"rel-{shape}",
                      tag.replace("distill-", "catalog-"), tag.replace("wide-", "catalog-"),
                      tag.replace("wide3-", "catalog-"),
                      # the fcall/staged arms share their size's architecture exactly, so they
                      # inherit its protocol measurement (the caption states this rule)
                      shape.replace("fcall-11m-big", "catalog-10m").replace("fcall-11m", "catalog-10m")
                           .replace("fcall-96m", "catalog-96m").replace("fcall-40m", "fcall-40m")):
        payload = report(candidate, "throughput")
        if payload:
            return payload["decode_tokens_per_second"]
    return None


_FT_METHOD = None


def ft_method() -> str:
    """Which fine-tuning every open row is scored under.

    Whichever of LoRA and full-parameter scores better, applied to every open row. The switch
    needs a full-parameter receipt for EVERY baseline (so the block is never half one method and
    half the other) AND a higher summed mean within both declared scope groups, because at a
    matched 1e-4 learning rate the
    full-parameter arm beat LoRA only on the 135M model and lost 8-11 points on everything larger.
    """
    global _FT_METHOD
    if _FT_METHOD is None:
        _FT_METHOD = "lora"
        # The unified arm is the -fullharness family: full parameters, the same 600-step budget
        # and the same union corpus as the LoRA rows, at 2e-5 rather than the 1e-4 that suits an
        # adapter. The earlier -full receipts were the learning-rate sweep and are not the block.
        if all((RESULTS / T1_DIR / f"ft-{tag}-fullharness.json").exists() for tag, _ in PUBLIC):
            def mean_of(prefix: str, tag: str, suites: tuple[str, ...]) -> float:
                values = [cell(f"{prefix}{tag}", suite, "type_match", T1_DIR)
                          for suite in suites]
                return sum(v for v in values if v is not None) / max(len(suites), 1)
            lora = [sum(mean_of("ft-", tag, suites) for tag, _ in PUBLIC)
                    for suites in (AUTHOR_CALL_SPLITS, STUDY_PROJECTIONS)]
            full = [sum(mean_of("ft-", f"{tag}-fullharness", suites)
                        for tag, _ in PUBLIC)
                    for suites in (AUTHOR_CALL_SPLITS, STUDY_PROJECTIONS)]
            # "the better one, applied to everybody" - not "whichever finished last". At a matched
            # 1e-4 the full-parameter arm was worse on every baseline above 135M, so this guard
            # keeps the paper on LoRA unless full actually wins.
            _FT_METHOD = "full" if all(f > l for f, l in zip(full, lora)) else "lora"
    return _FT_METHOD


def ft_tag(tag: str) -> str:
    """The fine-tuned receipt for an open model under whichever method the block is using."""
    return f"ft-{tag}-fullharness" if ft_method() == "full" else f"ft-{tag}"


def floors() -> dict[str, float]:
    """Majority-class rate per suite, taken from whichever baseline file covers all the suites."""
    for name in (f"{EVAL_DIR}/chance-baseline.json", f"{EVAL_DIR}/majority-baseline.json",
                 "evalsuite/chance-baseline.json"):
        path = RESULTS / name
        if path.exists():
            entries = json.loads(path.read_text())
            if all(suite in entries for suite in SUITES):
                return {suite: entry["majority_rate"] for suite, entry in entries.items()}
    return {}


def chance() -> dict[str, dict]:
    """Random choice among the row's own candidates — the floor that applies to an agent whose
    catalog arrives in the prompt. The majority-class floor answers a different question."""
    path = RESULTS / f"{EVAL_DIR}/chance-baseline.json"
    return json.loads(path.read_text()) if path.exists() else {}


def number(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}"


def row_html(cells: list[str], numeric_from: int = 1, bold: set[int] = frozenset(),
             css: str = "") -> str:
    parts = []
    for index, text in enumerate(cells):
        classes = (["n"] if index >= numeric_from else []) + (["best"] if index in bold else [])
        attribute = f' class="{" ".join(classes)}"' if classes else ""
        parts.append(f"<td{attribute}>{text}</td>")
    opening = f"<tr class='{css}'>" if css else "<tr>"
    return opening + "".join(parts) + "</tr>"


# Replicate spread per suite: the range across three seeds of one unchanged configuration. A
# difference smaller than this is not readable, so every claim in the ledger is checked against it.
T1_DIR = "evalsuite-full"   # all-rows protocol: every Table 1 cell scores the full pool

# Five runs of one unchanged configuration, four seeds between them.
REPLICATES = ("catalog-10m", "rc-full", "seed-101", "seed-202", "seed-303")


def spreads() -> dict[str, float]:
    """Replicate spread per suite, read from the receipts rather than pinned.

    A pinned table goes stale the moment the pool changes — the 200-row constants had BFCL at
    9.7 against a measured 17.0 and MobAct at 5.5 against 23.0 — and the spread is what every
    bold mark in the ledger is tested against. The floor is one row: a suite scored on 17 rows
    cannot resolve a difference finer than 5.9 points, whatever the replicates happen to do."""
    out = {}
    for suite in SUITES:
        scores, rows = [], []
        for tag in REPLICATES:
            payload = report(tag)
            row = (payload or {}).get("suites", {}).get(suite)
            if row and row.get("type_match") is not None:
                scores.append(row["type_match"] * 100)
                rows.append(row["rows"])
        if len(scores) < 3:
            out[suite] = 100.0
            continue
        out[suite] = round(max(max(scores) - min(scores), 100.0 / max(rows)), 1)
    return out


SPREAD = spreads()
GENERAL_SUITES = ("arc-easy", "arc-challenge", "hellaswag", "openbookqa", "winogrande", "mmlu")
GENERAL_TITLE = {"arc-easy": "ARC-e", "arc-challenge": "ARC-c", "hellaswag": "HellaSwag",
                 "openbookqa": "OBQA", "winogrande": "Wino", "mmlu": "MMLU"}


# --------------------------------------------------------------------------- tables

LEDGER_ARMS = [
    ("Import data into our model's SFT",
     "11M", "add the xLAM train split to the union corpus", "the data import",
     "catalog-10m", "wide-10m"),
    (None, "11M", "relabel that corpus with the strongest fine-tuned teacher",
     "teacher relabelling", "wide-10m", "distill-10m"),
    (None, "96M", "the two imports above together", "both together at 95.3M",
     "catalog-96m", "distill-96m"),
    (None, "96M", "add Toucan real-MCP trajectories [42]", "the Toucan import",
     "wide3-96m", "wide4-96m"),
    (None, "96M", "add schema-synth conversations from the ToolSandbox inventory",
     "the schema-synth import", "staged-fcall-96m", "staged-fcallts-96m"),
    (None, "11M", "add the ToolBench train split (70% name overlap)",
     "the ToolBench import", "distill-10m", "fctb-11m"),
    (None, "11M", "add the Mobile-Actions split its recipe had omitted",
     "the Mobile-Actions import", "distill-10m", "fcmob-11m"),
    (None, "96M", "add device-assistant data (0% exact name overlap)",
     "the stem-overlap import", "wide4-96m", "fcdev-96m"),
    (None, "11M", "all four splits above at once", "the combined import",
     "distill-10m", "fcall-11m"),
    (None, "96M", "all four splits above at once", "the combined import at 95.3M",
     "wide3-96m", "fcall-96m"),
    ("Import weights into our model",
     "11M", "initialise from a projected Qwen2.5-0.5B (cross-family)",
     "cross-family projection", "rc-scratch", "dc-qwen25-05b-blocks"),
    (None, "11M", "adopt an identical-shape sibling's weights (positive control)",
     "identical-shape adoption", "rc-scratch", "rc-full"),
    (None, "11M", "train under a donor's per-module learning-rate profile",
     "the adaptation profile", "distill-10m", "profiled-10m"),
    ("Scale the model",
     "11M&rarr;96M", "grow the architecture, recipe unchanged", "architecture scale",
     "catalog-10m", "catalog-96m"),
    (None, "96M", "pretraining budget 1,599 &rarr; 16,000 steps", "pretraining scale",
     "distill-96m", "arm2-96m"),
    (None, "96M", "coverage split and a stronger teacher on the 16,000-step backbone",
     "the stronger teacher", "arm2-96m", "arm3-96m"),
    (None, "11M", "pretraining budget 1,599 &rarr; 16,000 steps under the five-source SFT",
     "the 11M pretraining budget", "fcall-11m", "fcall-11m-big"),
    ("Swap the pretraining stream",
     "96M", "curated web &rarr; dialogue and distillation text, SFT matched",
     "the stream swap", "wide3-96m", "staged-wide3-96m"),
    (None, "96M", "the swapped stream under the five-source SFT",
     "the staged composition", "fcall-96m", "staged-fcall-96m"),
    (None, "11M", "the swapped stream under the five-source SFT at 10.5M",
     "the staged composition at 10.5M", "fcall-11m-big", "staged-fcall-11m"),
    (None, "43M", "the swapped stream under the five-source SFT at 43M",
     "the staged composition at 43M", "fcall-40m", "staged-fcall-40m")]
"""Every ledger intervention as (group, size, row label, caption name, control tag, treatment tag).

Hoisted out of table_ledger so the multiplicity audit scores the same rows the table renders rather than a second copy that could drift from it."""


SUITE_POOL = {"androidcontrol": 904, "mind2web": 252, "toolace": 949, "xlam": 2941,
              "bfcl": 204, "agentnet": 12482, "toolbench": 687, "toolsandbox": 17,
              "mcpatlas": 495, "mobileactions": 961}


def _multiplicity_audit() -> dict:
    """Score every band-bolded ledger cell against the Bonferroni threshold it claims to clear.

    The appendix used to assert that everything the band bolds would survive the correction. That
    is checkable, so it is checked here rather than asserted, and the sentence it feeds reports
    whatever this returns.
    """
    import math
    from statistics import NormalDist

    total = bolded = 0
    pending = []
    for arm in LEDGER_ARMS:
        control, treatment = arm[-2], arm[-1]
        for suite in SUITES:
            before = cell(control, suite, "type_match")
            after = cell(treatment, suite, "type_match")
            n, band = SUITE_POOL.get(suite), SPREAD.get(suite)
            if before is None or after is None or not n or band is None:
                continue
            total += 1
            # mirror table_ledger's bolding rule exactly: gains only, Mind2Web never bolded
            if suite == "mind2web" or (after - before) * 100 <= band:
                continue
            bolded += 1
            se = math.sqrt(before * (1 - before) / n + after * (1 - after) / n)
            z = abs(after - before) / se if se > 0 else float("inf")
            pending.append({"suite": suite, "delta": (after - before) * 100, "z": z,
                            "before": before * 100, "after": after * 100})
    critical = NormalDist().inv_cdf(1 - (0.05 / total) / 2)
    failures = [item for item in pending if item["z"] < critical]
    return {"comparisons": total, "bolded": bolded, "failures": failures, "critical": critical}


def _multiplicity_sentence() -> str:
    """Report the audit's own counts, so this paragraph cannot outlive the numbers behind it."""
    audit = _multiplicity_audit()
    failures = audit["failures"]
    if not failures:
        return (f"Of {audit['comparisons']} comparisons the band bolds {audit['bolded']}, and "
                f"every one clears z&nbsp;=&nbsp;{audit['critical']:.2f}.")
    by_suite: dict[str, list[dict]] = {}
    for item in failures:
        by_suite.setdefault(item["suite"], []).append(item)
    parts = []
    for suite, items in sorted(by_suite.items(), key=lambda kv: -len(kv[1])):
        worst = min(items, key=lambda item: item["z"])
        parts.append(f"{len(items)} on {SUITE_TITLE.get(suite, suite)} (smallest "
                     f"z&nbsp;=&nbsp;{worst['z']:.2f}, {worst['before']:.1f}&nbsp;&rarr;&nbsp;"
                     f"{worst['after']:.1f})")
    return (f"Of {audit['comparisons']} comparisons the band bolds {audit['bolded']}; "
            f"{audit['bolded'] - len(failures)} clear Bonferroni at "
            f"&alpha;&nbsp;=&nbsp;.05/{audit['comparisons']} "
            f"(z&nbsp;&gt;&nbsp;{audit['critical']:.2f}) and {len(failures)} do not: "
            + "; ".join(parts) + ". The MCP-Atlas failures sit where a base rate is near zero and "
            "the pool is small, so two or three points is large against the replicate band and "
            "small against the interval those rows admit; they are movement off a floor, not a "
            "measured effect. The remaining failures are near-threshold gains. All are bolded by "
            "our rule and should not be read as established.")


def _cost_scope() -> str:
    """What the cost measurement covers, and what it does not, read from the receipt itself.

    The second reviewer objection is partly that the cost column is one machine's decode rate
    presented as a device claim. Two of the quantities it asks for are already in the receipt and
    were simply not reported - prefill, which is time to first token, and the rate that fixes what
    a long catalog costs - so they are reported here; the rest are named as absent rather than
    left to be inferred.
    """
    path = RESULTS / "throughput" / "catalog-10m.json"
    try:
        ours = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return ""
    prefill_ms = ours["prefill_seconds"] * 1000
    prompt = ours["prompt_tokens"]
    rate = ours["prefill_tokens_per_second"]
    return (
        "<p><b>What the cost column measures, and what it does not.</b> One server CPU at four "
        "threads, fp16, a %d-token prompt and a 64-token answer, repeated for every row of "
        "Table 1 under one protocol. Two quantities come out of that run: "
        "decode throughput, which the table reports, and prefill, time to first token "
        "&mdash; %.1f&nbsp;ms for the 10.5M hybrid, at %s&nbsp;tokens/s. That rate is what a long "
        "catalog costs: the 33-schema ToolSandbox prompt, roughly seven times this one, is "
        "charged accordingly &mdash; a catalog-in-prompt cost paid "
        "before the first token. Not measured: resident memory, energy "
        "per decision, and the motivating hardware &mdash; no number here "
        "was taken on a phone, Raspberry Pi or embedded board &mdash; and the order-of-magnitude "
        "decode gap over a 0.5B model is a statement about this machine under this protocol, not "
        "a device measurement.</p>" % (prompt, prefill_ms, format(int(rate), ",")))


def _who_leads_numbers() -> dict[str, str]:
    """The open-block quantities the Who-leads paragraph quotes, read through ft_tag().

    These were typed, and had already drifted from the receipts under the current method (the
    prose said 41.6-67.0 where the receipts say 42.4-67.1, and credited six best marks where the
    receipts gave five). Reading them through ft_tag() also means the paragraph follows the
    LoRA/full switch instead of silently describing the method the block no longer uses.
    """
    def c(tag: str, suite: str):
        return cell(tag, suite, "type_match", T1_DIR)

    smol_pre = c("smollm2-135m", "xlam")
    smol_post = c(ft_tag("smollm2-135m"), "xlam")

    wins: dict[str, list[str]] = {}
    for suite in SUITES:
        best_tag, best_v = None, -1.0
        for tag, _ in PUBLIC + GENERIC:
            v = c(ft_tag(tag), suite)
            if v is not None and v > best_v:
                best_tag, best_v = tag, v
        if best_tag is not None:
            wins.setdefault(best_tag, []).append(suite)
    names = dict(PUBLIC) | dict(GENERIC)
    strongest = max(wins, key=lambda t: len(wins[t]))
    others = sorted(((t, ss) for t, ss in wins.items() if t != strongest),
                    key=lambda kv: -len(kv[1]))
    def short(tag: str) -> str:
        return names[tag].split(" [")[0]
    other_txt = "; ".join("%s takes %s" % (short(t), " and ".join(SUITE_TITLE.get(x, x) for x in ss))
                          for t, ss in others)

    tb = [c(ft_tag(t), "toolbench") for t, _ in PUBLIC]
    tb = [v * 100 for v in tb if v is not None]
    ac = []
    for t, _ in PUBLIC:
        a, b = c(t, "androidcontrol"), c(ft_tag(t), "androidcontrol")
        if a is not None and b is not None:
            ac.append((b - a) * 100)
    m2w = [c(ft_tag(t), "mind2web") for t, _ in PUBLIC]
    m2w += [c(tag, "mind2web") for tag, _, _, _ in OURS]
    m2w = [v * 100 for v in m2w if v is not None]

    ft_up = ft_total = 0
    all_improve = 0
    for suite in SUITES:
        deltas = [(c(ft_tag(t), suite), c(t, suite)) for t, _ in PUBLIC]
        deltas = [b - a for b, a in deltas if a is not None and b is not None]
        ft_total += len(deltas)
        ft_up += sum(1 for d in deltas if d > 0)
        all_improve += bool(deltas) and all(d > 0 for d in deltas)

    count_words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    return {
        "FTUP": str(ft_up), "FTTOT": str(ft_total),
        "FTALLN": count_words.get(all_improve, str(all_improve)),
        "SMOLPRE": f"{smol_pre * 100:.1f}", "SMOLPOST": f"{smol_post * 100:.1f}",
        "STRONGEST": short(strongest),
        "NBEST": count_words.get(len(wins[strongest]), str(len(wins[strongest]))) + " of the ten suites",
        "OTHERWINS": other_txt,
        "TBLO": f"{min(tb):.1f}", "TBHI": f"{max(tb):.1f}",
        "ACLO": f"{min(ac):+.1f}", "ACHI": f"{max(ac):+.1f}",
        "M2WLO": f"{min(m2w):.1f}", "M2WHI": f"{max(m2w):.1f}",
    }


def _fill(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("@" + key + "@", value)
    return text


def _generic_ladder_numbers() -> dict[str, str]:
    """Matched-stage summary for the generic-pretraining comparison.

    The shipped LocalAgent row has seen three additional SFT sources, so comparing it with the
    generic rows would confound pretraining with downstream inventory coverage. Keep both the
    main-text summary and the appendix discussion tied to the matched-stage receipt instead.
    """
    tags = {
        "P70": ft_tag("pythia-70m"),
        "P160": ft_tag("pythia-160m"),
        "P410": ft_tag("pythia-410m"),
        "DISTIL": ft_tag("distilgpt2"),
        "LOCAL": "baserecipe-96m",
        "SHIPPED": "fcall-96m",
    }

    def score(tag: str, suite: str) -> float:
        value = cell(tag, suite, "type_match", T1_DIR)
        if value is None:
            raise RuntimeError(f"missing matched-stage receipt: {tag}/{suite}")
        return value * 100

    def mean(tag: str, suites: tuple[str, ...]) -> float:
        return sum(score(tag, suite) for suite in suites) / len(suites)

    values: dict[str, str] = {}
    for label, tag in tags.items():
        values[f"G{label}AUTHOR"] = f"{mean(tag, AUTHOR_CALL_SPLITS):.1f}"
        values[f"G{label}PROJECTED"] = f"{mean(tag, STUDY_PROJECTIONS):.1f}"

    for label, suite in (("TA", "toolace"), ("XLAM", "xlam"), ("BFCL", "bfcl"),
                         ("TB", "toolbench"), ("AC", "androidcontrol")):
        delta = score(tags["LOCAL"], suite) - score(tags["P160"], suite)
        values[f"G{label}DELTA"] = (f"+{delta:.1f}" if delta >= 0
                                     else f"&minus;{abs(delta):.1f}")

    values["GLOCALTB"] = f"{score(tags['LOCAL'], 'toolbench'):.1f}"
    values["GTBCHANCE"] = f"{chance()['toolbench']['random_within_catalog'] * 100:.1f}"
    return values


def _region_gain(tag: str, suites: tuple[str, ...]) -> float:
    """Mean function-name gain over the same-shape random-init control, in points."""
    values = []
    for suite in suites:
        arm = cell("rc-" + tag, suite, "type_match", T1_DIR)
        base = cell("rc-scratch", suite, "type_match", T1_DIR)
        if arm is None or base is None:
            raise RuntimeError(f"missing same-shape region receipt: {tag}/{suite}")
        values.append((arm - base) * 100)
    return sum(values) / len(values)


def _region_gain_numbers() -> dict[str, str]:
    values = {}
    for label, tag in (("ATTN", "attn"), ("FFN", "ffn"), ("FULL", "full")):
        values[f"RG{label}AUTHOR"] = f"{_region_gain(tag, AUTHOR_CALL_SPLITS):.1f}"
        values[f"RG{label}PROJECTED"] = f"{_region_gain(tag, STUDY_PROJECTIONS):.1f}"
    values["RGATTNAUTHORSHARE"] = (
        f"{100 * _region_gain('attn', AUTHOR_CALL_SPLITS) / _region_gain('full', AUTHOR_CALL_SPLITS):.0f}")
    values["RGATTNPROJECTEDSHARE"] = (
        f"{100 * _region_gain('attn', STUDY_PROJECTIONS) / _region_gain('full', STUDY_PROJECTIONS):.0f}")
    return values


def table_ledger() -> str:
    """Every intervention as a delta against its own control, read against the noise floor.

    This is the table the report exists to produce: three candidate answers to "how do I get an
    agent onto my device" — import data, import weights, or design the architecture — each priced
    on the same suites against the control that isolates it.
    """
    # (label, control tag, treatment tag) — each pair varies exactly one thing.
    # (row label, name the caption uses, control tag, treatment tag)
    # (group, model size, row label, caption short-name, control tag, treatment tag).
    # Every row intervenes on the custom model; open models appear only as donors or teachers.
    arms = LEDGER_ARMS

    # Which arms clear their suite's spread is a property of the rows, not of a sentence someone
    # typed once; recompute it so the caption cannot outlive the numbers.
    def cleared(control: str, treatment: str) -> list[str]:
        names = []
        for suite in SUITES:
            before, after = cell(control, suite, "type_match"), cell(treatment, suite, "type_match")
            if before is None or after is None or suite == "mind2web":
                continue
            if abs((after - before) * 100) > SPREAD[suite]:
                names.append(SUITE_TITLE[suite])
        return names

    moved = [short for _, _, _, short, control, treatment in arms if cleared(control, treatment)]
    # AgentNet's spread is wide enough that naming which rows survive it must be computed, not typed.
    on_agentnet = [short for _, _, _, short, control, treatment in arms
                   if SUITE_TITLE["agentnet"] in cleared(control, treatment)]
    still = [short for _, _, _, short, control, treatment in arms if not cleared(control, treatment)]
    join = lambda names: ", ".join(names[:-1]) + f" and {names[-1]}"
    clears_sentence = (f"Outside Mind2Web, {len(moved)} of the {len(moved) + len(still)} "
                       f"interventions clear the spread somewhere ({join(moved[:4])} among "
                       f"them) and the remaining {len(still)}, including {join(still[:3])}, "
                       f"clear it on no suite.")
    # Every figure the caption quotes is read back out of the same receipts the rows are, so the
    # sentence cannot drift from the table when the pool or the replicate set changes.
    adoption = ((cell("rc-full", "agentnet", "type_match")
                 - cell("rc-scratch", "agentnet", "type_match")) * 100)
    caption = ("Function-name-match change against each row's stated control. Bold marks an "
               "absolute change that exceeds the "
               "measured replicate spread (five runs, four seeds); Mind2Web is floor-dominated "
               "and never bolded. Open models appear only as donors or teachers. Full noise and "
               "multiplicity analysis: App. A.")
    header = ("<thead><tr><th>Model</th><th>Intervention (vs our model's own control)</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES)
              + "</tr></thead>")
    rows = []
    for group, model, label, _short, control, treatment in arms:
        if group:
            rows.append(f"<tr class='group'><td colspan='{2 + len(SUITES)}'>{group}</td></tr>")
        cells = [model, label]
        for suite in SUITES:
            before, after = cell(control, suite, "type_match"), cell(treatment, suite, "type_match")
            if before is None or after is None:
                cells.append("—")
                continue
            delta = (after - before) * 100
            # Mind2Web is floor-dominated: the majority class answers 79% and every model sits
            # within the replicate band of that floor, so a delta there is never emphasised.
            significant = abs(delta) > SPREAD[suite] and suite != "mind2web"
            cells.append(f"<b>{delta:+.1f}</b>" if significant else f"{delta:+.1f}")
        rows.append(row_html(cells, numeric_from=2))
    rows.append(row_html(["", "replicate spread (noise band, 5 runs, 4 seeds)"]
                         + [f"±{SPREAD[s]:.1f}" if SPREAD[s] < 100 else "—" for s in SUITES],
                         css="floor"))
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def mean_cell(tags: list[str], suite: str) -> float | None:
    """Mean over the arms that have reported, so a band is read across donors rather than one."""
    values = [cell(tag, suite, "type_match") for tag in tags]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def table_transfer() -> str:
    """Which depth, and which donor size — the anatomy of weight transfer, compactly.

    Scored on the open-catalog suites only: those are the suites where the function name is
    informative, and a depth effect that shows up nowhere else is not one.
    """
    caption = ("Weight transfer taken apart, on the seven open-catalog suites. <em>Depth</em> "
               "transfers one band and leaves the rest random, so results attribute to a depth; "
               "cross-family bands average a 0.6B and a 350M donor (per-donor receipts ship with the release). "
               "<em>Size ratio</em> sweeps the donor from 1.4&times; to 6.3&times; into the "
               "fixed 95.3M model; the same-family 1.5&times; arm crosses only the shape. A "
               "projected arm substitutes for pretraining, so its control is random init "
               "<em>without</em> pretraining and none beats it; the pretrained chain in the "
               "last row is the ceiling the substitution had to reach. Function-name match (%).")
    donors = ("qwen3-06b", "lfm2-350m")
    depth = [("first half (same family, 10.5M)", ["rc-early"]),
             ("second half (same family, 10.5M)", ["rc-late"]),
             ("whole backbone (pretrained same-shape sibling, 10.5M)", ["rc-full"]),
             ("early third (cross-family, 95.3M)", [f"band-{d}-early" for d in donors]),
             ("middle third (cross-family, 95.3M)", [f"band-{d}-middle" for d in donors]),
             ("late third (cross-family, 95.3M)", [f"band-{d}-late" for d in donors])]
    ratios = [("SmolLM2-135M (1.4&times;)", ["ratio-96m-smollm2-135m"]),
              ("LFM2.5-230M (2.4&times;)", ["ratio-96m-lfm25-230m"]),
              ("LFM2-350M (3.7&times;)", ["ratio-96m-lfm2-350m"]),
              ("Qwen3-0.6B (6.3&times;)", ["ratio-96m-qwen3-06b"]),
              ("own 16M, same family (1.5&times;)", ["ratio-10m-own16m"])]
    controls = [("random init, no pretraining (10.5M)", ["rc-scratch"]),
                ("random init + full pretraining (95.3M)", ["catalog-96m"])]
    groups = [("Which layers are transferred", depth),
              ("How much bigger the donor is", ratios),
              ("Floor and ceiling", controls)]
    header = ("<thead><tr><th>Arm</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in OPEN_CATALOG)
              + "</tr></thead>")
    rows = []
    for title, group in groups:
        present = [(name, tags) for name, tags in group
                   if any(report(tag) is not None for tag in tags)]
        if not present:
            continue
        rows.append(f"<tr class='group'><td colspan='{1 + len(OPEN_CATALOG)}'>{title}</td></tr>")
        for name, tags in present:
            rows.append(row_html([name] + [number(mean_cell(tags, s)) for s in OPEN_CATALOG]))
    if not rows:
        return ""
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_transfer_full() -> str:
    """Every transfer arm on every suite, for the appendix."""
    caption = ("Every weight-transfer arm on all suites, including the per-donor depth bands "
               "the main text averages. The depth-half rows use the 10.5M same-family model; "
               "the band and ratio rows project into the 95.3M model. Function-name match (%).")
    donors = ("qwen3-06b", "lfm2-350m")
    arms = [(f"band-{d}-{b}", f"{d}, {b} third") for d in donors
            for b in ("early", "middle", "late")]
    arms += [("ratio-96m-smollm2-135m", "SmolLM2-135M donor (1.4&times;)"),
             ("ratio-96m-lfm25-230m", "LFM2.5-230M donor (2.4&times;)"),
             ("ratio-96m-lfm2-350m", "LFM2-350M donor (3.7&times;)"),
             ("ratio-96m-qwen3-06b", "Qwen3-0.6B donor (6.3&times;)"),
             ("ratio-10m-own16m", "own 16M donor, same family (1.5&times;)"),
             ("rc-scratch", "random initialisation (10.5M)"),
             ("catalog-96m", "random initialisation (95.3M)")]
    rows = [row_html([name] + [number(cell(tag, s, "type_match")) for s in SUITES])
            for tag, name in arms if report(tag) is not None]
    if not rows:
        return ""
    header = ("<thead><tr><th>Arm</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES) + "</tr></thead>")
    return (f'<table data-caption="{caption}" class="long">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_layers() -> str:
    """Where fine-tuning concentrates, by depth third, for every family — and across scales."""
    path = RESULTS / "analysis/layer_profiles.json"
    if not path.exists():
        return ""
    profiles = json.loads(path.read_text())
    scale_path = RESULTS / "analysis/layer_profiles_scale.json"
    if scale_path.exists():
        profiles.update(json.loads(scale_path.read_text()))
    order = [("smollm2-135m", "SmolLM2-135M"), ("lfm25-230m", "LFM2.5-230M"),
             ("lfm2-350m", "LFM2-350M"), ("granite-h-350m", "Granite-4.0-H-350M"),
             ("granite-350m", "Granite-4.0-350M"), ("smollm2-360m", "SmolLM2-360M"),
             ("danube3-500m", "danube3-500M"), ("qwen25-05b", "Qwen2.5-0.5B"),
             ("qwen25-coder-05b", "Qwen2.5-Coder-0.5B"), ("lfm2-700m", "LFM2-700M"),
             ("qwen3-06b", "Qwen3-0.6B"), ("qwen3-17b", "Qwen3-1.7B"),
             ("qwen3-4b", "Qwen3-4B")]
    rows = []
    for tag, name in order:
        got = profiles.get(tag)
        if not got:
            continue
        curve = got["per_layer_mean"]
        n = len(curve)
        third = max(1, n // 3)
        first = sum(curve[:third]) / third * 1e3
        middle = sum(curve[third:2 * third]) / third * 1e3
        last = sum(curve[2 * third:]) / max(1, n - 2 * third) * 1e3
        peak = max(range(n), key=lambda i: curve[i])
        rows.append(row_html([name, str(n), f"{first:.1f}", f"{middle:.1f}", f"{last:.1f}",
                              f"{(peak + 0.5) / n:.2f}"]))
    if not rows:
        return ""
    caption = ("Where the shared agent recipe moves each model: mean per-layer ‖dW‖/‖W‖ "
               "(&times;10³) by depth third, and the depth of the single most-moved layer as a "
               "fraction of the stack. The set spans GQA transformers, LFM2 gated-convolution "
               "hybrids and Granite-H mamba hybrids at 135M–4B; agreement across them is "
               "evidence the concentration belongs to the task, not to an architecture.")
    header = ("<thead><tr><th>Model</th><th class='n'>layers</th>"
              "<th class='n'>first third</th><th class='n'>middle</th>"
              "<th class='n'>last third</th><th class='n'>peak depth</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_errors() -> str:
    """Failure modes decomposed: where each error family lives, and what removes it."""
    arms = [("errors-distill-96m", "LocalAgent-96M, 1,599-step budget"),
            ("errors-arm2-96m", "LocalAgent-96M, 16,000-step budget"),
            ("errors-ft-qwen3-06b", "ft-Qwen3-0.6B (reference)")]
    rows = []
    for tag, name in arms:
        payload = report(tag, "analysis")
        if payload is None:
            continue
        rows.append(f"<tr class='group'><td colspan='5'>{name}</td></tr>")
        for suite, row in payload["suites"].items():
            shares = row["shares"]
            rows.append(row_html([SUITE_TITLE.get(suite, suite),
                                  f"{shares['correct_name']:.1f}",
                                  f"{shares['wrong_in_catalog']:.1f}",
                                  f"{shares['out_of_catalog']:.1f}",
                                  f"{shares['parse_fail']:.1f}"]))
    if not rows:
        return ""
    caption = ("Errors on the open-catalog suites decomposed by mechanism: wrong choice within "
               "the offered catalog, a tool named that the row never offered, and parse failure. "
               "Raising the pretraining budget 10&times; halves out-of-catalog emission on BFCL "
               "and cuts it by a third on ToolBench while discrimination barely moves; the "
               "fine-tuned open reference has eliminated the format family entirely and cut the "
               "reading family to 0.5&ndash;2.0%, leaving discrimination as nearly the whole remainder.")
    header = ("<thead><tr><th>Suite</th><th class='n'>correct</th><th class='n'>wrong "
              "in-catalog</th><th class='n'>out-of-catalog</th><th class='n'>parse fail</th>"
              "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_scale() -> str:
    """One family, three scales, identical recipe — released and fine-tuned."""
    rows = []
    for tag, name in (("qwen3-06b", "Qwen3-0.6B"), ("qwen3-17b", "Qwen3-1.7B"),
                      ("qwen3-4b", "Qwen3-4B")):
        for arm, label in ((tag, "as released"), (f"ft-{tag}", "fine-tuned")):
            if report(arm) is None:
                continue
            rows.append(row_html([f"{name}, {label}"]
                                 + [number(cell(arm, s, "type_match")) for s in SUITES]))
    if len(rows) < 3:
        return ""
    caption = ("The scale series: one family under the identical recipe at 0.6B, 1.7B and 4B. "
               "Function-name match (%); the corresponding per-layer profiles are in the depth-profile "
               "table above.")
    header = ("<thead><tr><th>Model</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES) + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_causal() -> str:
    """Adapting only the top-dW band versus only the bottom band, against the all-layer arm."""
    models = [("smollm2-135m", "SmolLM2-135M"), ("lfm2-350m", "LFM2-350M"),
              ("qwen3-06b", "Qwen3-0.6B")]
    rows, present = [], False
    for tag, name in models:
        arms = [(tag, "as released"), (f"band-ft-{tag}-top", "top-‖dW‖ third only"),
                (f"band-ft-{tag}-bottom", "bottom third only"), (f"ft-{tag}", "all layers")]
        have = [(t, label) for t, label in arms if report(t) is not None]
        if len(have) < 3:
            continue
        present = True
        rows.append(f"<tr class='group'><td colspan='{1 + len(SUITES)}'>{name}</td></tr>")
        for t, label in have:
            rows.append(row_html([label]
                                 + [number(cell(t, suite, "type_match")) for suite in SUITES]))
    if not present:
        return ""
    caption = ("The causal check: LoRA restricted to the top third of layers ranked by that "
               "model's own measured ‖dW‖/‖W‖, versus the bottom third, versus every layer; "
               "same recipe, same steps, one third the adapted layers in each band arm. If "
               "agentic ability concentrates where the profile says, the top band recovers most "
               "of the all-layer gain and the bottom band little. Function-name match (%).")
    header = ("<thead><tr><th>Adapted layers</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES) + "</tr></thead>")
    return (f'<table data-caption="{caption}" class="long">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_answers() -> str:
    """The four questions the report set out to answer, each with its evidence pointer."""
    caption = ("The practitioner's questions, answered against the replicate noise floor; "
               "each row points at the section and table that establishes it.")
    rows = [
        ("<b>Design the architecture at all?</b>",
         "Only when latency excludes 350M: it buys 604 tok/s, not controlled accuracy. The "
         "available general-capability audit covers the earlier LoRA reference, not the current "
         "full-parameter stage (§3, §6, App. C)"),
        ("<b>Q1</b>: import open data?",
         "Yes: the two data channels are the only transfers clearing the noise floor; they buy "
         "inventory, not catalog skill (§4, T3)"),
        ("<b>Q2</b>: import open weights?",
         "Not by slicing: every shape-crossing arm hits the floor, even same-family; "
         "identical-shape adoption is the positive control, not a route (§4, App. B)"),
        ("<b>Q3</b>: which layers hold ability?",
         "None exclusively: attention adoption retains most, feed-forward none, yet fine-tuning "
         "writes feed-forward hardest; the least-written third recovers 85% of the gain (§4, App. B)"),
        ("<b>Q4</b>: how is any of it measured honestly?",
         "One harness, one prompt contract and parser for every model; chance-in-catalog floors, "
         "not majority class; replicate spreads gate every claim (§3, App. B–D)"),
    ]
    body = "".join(row_html([q, a], numeric_from=99) for q, a in rows)
    header = ("<thead><tr><th></th><th>Answer</th></tr></thead>")
    return (f'<table data-caption="{caption}" class="answers">' + header + "<tbody>"
            + body + "</tbody></table>")


def _episode_corpus_rows() -> list[tuple[str, str, str, str]]:
    """The episode supervision splits for the data table, from the measured receipt.

    The corpus behind the abstract's episode numbers was absent from the table that promises to
    record every split each stage consumes. Counts, plan lengths and token totals are measured
    (results/analysis/episode_corpus_stats.json), not quoted.
    """
    path = RESULTS / "analysis/episode_corpus_stats.json"
    try:
        stats = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return []
    rows = []
    for key, label, role in (
            ("agentic-free", "episode decisions (base)", "agentic arms' episode split (App. E)"),
            ("agentic-free-xl", "episode decisions (4&times;)",
             "the 4&times; arms' split: new tasks, error injection 0.45 vs 0.35")):
        entry = stats.get(key)
        if not entry:
            continue
        detail = ("%s generated tasks, 7 families balanced, gold plans mean %.1f / max %d "
                  "steps, %.2fM tokens; %s recovery and %s abstention decisions"
                  % (format(entry["tasks"], ","), entry["plan_mean"], entry["plan_max"],
                     entry["tokens"] / 1e6, format(entry["by_kind"]["recovery"], ","),
                     format(entry["by_kind"]["abstain"], ",")))
        rows.append((label, role, format(entry["rows"], ","), detail))
    return rows


def table_train_sources() -> str:
    """Every conversation split the midtrain and SFT stages consume."""
    rows_data = [
        ("merged-v2 union", "midtrain agent mix; the shared SFT corpus", "16,300",
         "3 public corpora + 9.9% synthetic multi-turn conversations; not simulator decisions"),
        ("distill2 train-clean", "the union, teacher-relabelled (§4)", "40,812",
         "teacher call kept only on gold agreement"),
        ("xLAM split", "§4 data-import arm", "18,000", "Salesforce xlam-function-calling-60k [6]"),
        ("Mobile-Actions train", "the ladder's device split", "8,692", "official train split [41]"),
        ("ToolBench train", "import arm, 70% eval-name overlap", "6,000",
         "from the 46,886-row ToolLLaMA split [17]"),
        ("Toucan calls", "diverse real-MCP import", "5,911", "single-turn originals [42]"),
        ("r0b0t calls", "staged midtrain agentic split", "4,845", "tool rows of [45]"),
        ("device-assistant", "stem-overlap control (0% exact)", "2,251", "API-Bank train + ToolTalk scenarios"),
        ("ToolSandbox schema-synth", "same-inventory arm, disclosed", "1,066",
         "from the 33 public schemas; no test scenario read"),
    ]
    rows_data += _episode_corpus_rows()
    caption = ("All splits pass the trainer contract (schema whitelist, gold-argument "
               "conformance, duplicate-name and marker rejection, real-tokenizer length) and "
               "are deduplicated against the evaluation pools at the rendered-prompt level. "
               "Episode rows are one decision each, their user turn byte-identical to the "
               "runtime's observation; the 4&times; "
               "corpus is newly generated tasks at a higher error-injection rate, not the base "
               "corpus repeated, and its evaluation split is unchanged. "
               "Together the three tables cover every token any stage consumes.")
    header = ("<thead><tr><th>Split</th><th>Role</th><th class='n'>rows</th>"
              "<th>Source and filter</th></tr></thead>")
    body = [row_html(list(r), numeric_from=2) for r in rows_data]
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(body) + "</tbody></table>")


def table_pretrain_sources() -> str:
    """Both pretraining streams' construction, from the shard manifests."""
    path = RESULTS / "analysis/pretrain_manifests.json"
    if not path.exists():
        return ""
    manifests = json.loads(path.read_text())
    NAME = {"mixture:fineweb_edu_dedup": ("fineweb-edu (deduplicated)", "educational web text", "ODC-BY-1.0"),
            "mixture:cosmopedia_v2": ("cosmopedia-v2", "synthetic encyclopedia and textbooks", "Apache-2.0"),
            "mixture:permissive_python": ("permissive-python", "permissively licensed code", "MIT/BSD/Apache mix"),
            "mixture:ultrachat": ("UltraChat", "open multi-turn dialogue", "MIT"),
            "mixture:hh_rlhf": ("hh-rlhf", "helpfulness dialogue", "MIT"),
            "mixture:hh-rlhf": ("hh-rlhf", "helpfulness dialogue", "MIT"),
            "mixture:kimi-k3-distill": ("kimi-k3 traces [45]", "frontier distillation traces", "open release"),
            "mixture:qwen38-50k": ("qwen3.8-max traces [45]", "frontier distillation traces", "open release"),
            "mixture:r0b0t-general": ("r0b0t general [45]", "pooled distillation text", "open release"),
            "mixture:manus-distill": ("manus traces [45]", "agent-style traces", "open release")}
    caption = ("Construction of the two pretraining streams, from their shard manifests. "
               "<em>pt-big</em> is the curated stream behind the Table 1 ladder; "
               "<em>pool-general</em> is the open-dataset stream behind the staged backbone of "
               "App. C, assembled entirely from openly released dialogue and distillation "
               "corpora. Both are packed once, with near-deduplication and a rendered-prompt "
               "evaluation denylist, into 2,048-token rows under our models' 16k vocabulary; "
               "the Table 1 budgets consume the same stream for 1,599 or 16,000 steps, so "
               "budget comparisons vary optimisation, never the text. Per-document licence "
               "counts ship in the manifests.")
    header = ("<thead><tr><th>Source</th><th class='n'>documents</th>"
              "<th class='n'>share</th><th>licence</th></tr></thead>")
    rows = []
    for stream, label in [("pt-big", "pt-big — the curated stream (Table 1 ladder)"),
                          ("pool-general", "pool-general — the open-dataset stream (staged backbone)")]:
        manifest = manifests.get(stream)
        if not manifest:
            continue
        counts = manifest["source_counts"]
        total = sum(counts.values())
        rows.append(f"<tr class='group'><td colspan='4'>{label}</td></tr>")
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            name, content, licence = NAME.get(key, (key.split(":")[-1], "", ""))
            source = f"{name} &mdash; {content}" if content else name
            rows.append(row_html([source, f"{n:,}", f"{100 * n / total:.0f}%", licence],
                                 numeric_from=1))
        rows.append(row_html(["<b>total</b>", f"<b>{total:,}</b>", "", ""], numeric_from=1))
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_pretrain_corpus() -> str:
    """Five open pretraining corpora at one step budget, priced in tokens and wall clock."""
    arms = [("qwen38-50k", "Qwen3.8-Max distillation (50k)"),
            ("manus-distill", "Manus multi-teacher distillation"),
            ("hh-rlhf", "Anthropic HH-RLHF"),
            ("kimi-k3-distill", "Qwen3.8/GLM5.2/Kimi-K3 distillation"),
            ("ultrachat", "UltraChat-200k")]
    rows = []
    for tag, name in arms:
        corpus = report(f"{tag}.txt", "pretrain-manifests")
        payload = report(f"pc-{tag}")
        if corpus is None and payload is None:
            continue
        cells = [name]
        cells += ["—" if corpus is None else f"{corpus['documents'] / 1e3:.0f}k",
                  "—" if corpus is None else f"{corpus['tokens'] / 1e6:.0f}",
                  "—" if corpus is None else f"{corpus['chars_per_token']:.2f}"]
        seconds = (corpus or {}).get("pretrain_seconds")
        cells.append("—" if seconds is None else f"{seconds / 60:.0f}")
        cells += [number(cell(f"pc-{tag}", suite, "type_match")) for suite in OPEN_CATALOG]
        rows.append(row_html(cells))
    if not rows:
        return ""
    caption = ("Five open corpora as the pretraining stage, packed into the same shard layout "
               "and run through the identical chain at the same 1,599-step budget: every "
               "arm consumes the same 209.7M tokens, the corpus size below is what the budget "
               "samples from, not what it reads, and implied epochs vary by two orders of "
               "magnitude. "
               "Function-name match (%) after the shared post-training; each pretraining "
               "stage took 20–21 minutes on one otherwise idle H100 NVL, sequentially. "
               "A packing fault made all five arms consume the identical original token stream "
               "(retraction below the table): read these rows as replicates, not a "
               "corpus comparison.")
    header = ("<thead><tr><th>Pretraining corpus</th><th class='n'>docs</th>"
              "<th class='n'>tokens (M)</th><th class='n'>chars/tok</th>"
              "<th class='n'>pretrain (min)</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in OPEN_CATALOG)
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_general() -> str:
    """What the agent recipe costs the model everywhere else."""
    caption = ("General-suite change after the earlier LoRA reference recipe (percentage "
               "points; 300 identical rows before/after per suite), scored by length-normalised "
               "continuation likelihood. Mean is unweighted across the six suites; absolute "
               "scores and chance floors are in App. C. These receipts point to "
               "<span class='mono'>runs/lora/*</span>; they do not measure Table 1's current "
               "full-parameter 2e-5 stage.")
    header = ("<thead><tr><th>Model</th>"
              + "".join(f"<th class='n'>{GENERAL_TITLE[s]}</th>" for s in GENERAL_SUITES)
              + "<th class='n'>mean &Delta;</th></tr></thead>")
    rows = []
    for tag, name in PUBLIC:
        payload = report(tag, "general")
        if payload is None or "delta" not in payload:
            continue
        deltas = payload["delta"]
        rows.append(row_html(
            [name] + [f"{deltas[s]:+.1f}" if s in deltas else "—" for s in GENERAL_SUITES]
            + [f"<b>{payload['mean_delta']:+.2f}</b>"]))
    if not rows:
        return ""
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_gaia() -> str:
    """GAIA closed-book lower bound, released and fine-tuned."""
    order = [("qwen3-06b", "Qwen3-0.6B"), ("qwen25-05b", "Qwen2.5-0.5B"),
             ("smollm2-360m", "SmolLM2-360M"), ("granite-h-350m", "Granite-4.0-H-350M"),
             ("lfm2-700m", "LFM2-700M")]
    rows = []
    for tag, name in order:
        for arm, label in ((tag, "released"), (f"ft-{tag}", "fine-tuned")):
            payload = report(arm, "gaia")
            if payload is None:
                continue
            levels = payload["per_level"]
            rows.append(row_html([f"{name}, {label}", f"{payload['accuracy']:.1f}",
                                  f"{levels.get('1', 0):.1f}", f"{levels.get('2', 0):.1f}",
                                  f"{levels.get('3', 0):.1f}"]))
    for arm, name in (("ft-qwen3-17b", "Qwen3-1.7B, fine-tuned"),
                      ("ft-qwen3-4b", "Qwen3-4B, fine-tuned")):
        payload = report(arm, "gaia")
        if payload:
            levels = payload["per_level"]
            rows.append(row_html([name, f"{payload['accuracy']:.1f}",
                                  f"{levels.get('1', 0):.1f}", f"{levels.get('2', 0):.1f}",
                                  f"{levels.get('3', 0):.1f}"]))
    if not rows:
        return ""
    caption = ("GAIA validation as a stated lower bound: the 127 text-only questions (attachment "
               "rows excluded), closed book, no browsing or tools, GAIA-normalised exact match "
               "on the final answer, not a GAIA leaderboard score. Accuracy (%) overall and by "
               "level.")
    header = ("<thead><tr><th>Model</th><th class='n'>overall</th><th class='n'>L1</th>"
              "<th class='n'>L2</th><th class='n'>L3</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_general_absolute() -> str:
    """The same measurement in absolute terms, for the supplement."""
    caption = ("General-capability accuracy (%) before and after the earlier LoRA reference "
               "recipe, absolute. These receipts are not the current full-parameter Table 1 stage. "
               "Chance is 25% on ARC, OpenBookQA and MMLU, 25% on HellaSwag and 50% on "
               "Winogrande, so a model near those values had little to lose on that suite.")
    header = ("<thead><tr><th>Model</th><th>Weights</th>"
              + "".join(f"<th class='n'>{GENERAL_TITLE[s]}</th>" for s in GENERAL_SUITES)
              + "</tr></thead>")
    rows = []
    for tag, name in PUBLIC:
        payload = report(tag, "general")
        if payload is None:
            continue
        for stage, label in (("base", "as released"), ("finetuned", "fine-tuned")):
            scores = payload["scores"].get(stage)
            if not scores:
                continue
            rows.append(row_html([name if stage == "base" else "", label]
                                 + [number(scores[s] / 100) if s in scores else "—"
                                    for s in GENERAL_SUITES], numeric_from=2))
    if not rows:
        return ""
    return (f'<table data-caption="{caption}" class="long">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def _open_ft_description() -> str:
    """The open block's fine-tuning, described as it is actually scored, per the method gate."""
    if ft_method() == "full":
        return (
            "Open baselines take full-parameter updates: AdamW 2e-5 with "
            "warmup-decay, batch 8 at maximum length 1,024, 600 steps on the union "
            "corpus, gradient steps skipped on non-finite norms. The adapter alternative &mdash; "
            "LoRA [7] at rank 16 (alpha 32, dropout 0.05) on every attention and feed-forward "
            "projection, AdamW at 1e-4 &mdash; sums lower over "
            "the block (table below), so the gate selects full parameters; 1e-4 is a normal adapter "
            "rate, roughly five times a normal full-parameter one: each method runs "
            "at its own rate. One scoring exception: "
            "Granite-4.0-H is <em>scored</em> at batch 4, its mamba2 path materialising a "
            "six-index tensor whose 32-row batch requests 144&nbsp;GiB.")
    return (
        "Open baselines take LoRA [7] at rank 16 (alpha 32, dropout 0.05) on every attention and "
        "feed-forward projection: AdamW at 1e-4 with warmup-decay scheduling, batch 8 at maximum "
        "length 1,024, 600 steps, gradient steps skipped on non-finite norms.")


def _adapter_clause() -> str:
    if ft_method() == "full":
        return ", the update type every row in Table 1 now shares"
    return " rather than the baselines' adapters"


def _rate_twin() -> str:
    """The control at the block's own 2e-5, per size, once its receipts exist; silent until then."""
    parts = []
    for size in ("96m", "40m"):
        twin = report(f"baserecipe2e5-{size}", T1_DIR)
        base = report(f"baserecipe-{size}", T1_DIR)
        if twin is None or base is None:
            continue
        tag_twin, tag_base = f"baserecipe2e5-{size}", f"baserecipe-{size}"
        a0 = scope_mean(tag_base, AUTHOR_CALL_SPLITS, directory=T1_DIR)
        a1 = scope_mean(tag_twin, AUTHOR_CALL_SPLITS, directory=T1_DIR)
        p0 = scope_mean(tag_base, STUDY_PROJECTIONS, directory=T1_DIR)
        p1 = scope_mean(tag_twin, STUDY_PROJECTIONS, directory=T1_DIR)
        parts.append("%s author %.1f&rarr;%.1f, projection %.1f&rarr;%.1f" %
                     ("95.3M" if size == "96m" else "42.8M", a0, a1, p0, p1))
    if not parts:
        return ""
    return (" Re-running at the block's own 2e-5 gives %s. The effect is mixed across "
            "scope and size; the rate-asymmetry table below reports it rather than pooling it."
            % "; ".join(parts))


def _method_tokens() -> dict[str, str]:
    """One source of truth for how each block was fine-tuned, filled per the method gate.

    The gate flip left six hand-written descriptions of the open block disagreeing with each
    other on the same build. Every prose description of the block's method now comes from here,
    and the per-row ledger is table_recipes.
    """
    if ft_method() == "full":
        return {
            "OPENRECIPE": ("full-parameter updates, AdamW at 2e-5, batch 8, 600 steps (the "
                           "adapter alternative is priced in App. B)"),
            "FAIRNESSLEAD": (
                "Our models train full-parameter because this trainer has no adapter path, "
                "and the method gate has since selected full parameters for the open block, so "
                "update type is matched across Table 1; unmatched are learning rate "
                "(2e-4 under WSD against the block's 2e-5) and the budget; the table below "
                "prices the choice."),
            "BASERECIPENOTE": (" That stage is the block's LoRA-era recipe; Table 1's "
                               "block has since moved to full parameters at 2e-5, so this "
                               "control matches its update type and differs in rate."),
            "UNMATCHED": "",
        }
    return {
        "OPENRECIPE": ("LoRA [7] r=16 on attention and feed-forward projections, AdamW at 1e-4, "
                       "batch 8, 600 steps"),
        "FAIRNESSLEAD": (
            "Our own models are trained with full-parameter updates because this trainer has no "
            "adapter path, while every open row is fine-tuned with LoRA, so one objection is that "
            "the open block is scored under the weaker method."),
        "BASERECIPENOTE": "",
        "UNMATCHED": "full-parameter where the baselines use LoRA, and",
    }


def table_recipes() -> str:
    """Which configuration produced every row family in this report, in one place.

    Six prose descriptions of the fine-tuning drifted apart when the method gate flipped; a
    reader asking 'which optimiser and rate made this checkpoint' should not reconstruct it from
    scattered sentences. Values here are the training configs' constants, and the open-block row
    follows the gate.
    """
    full = ft_method() == "full"
    open_row = (["Open block (11 models, Table 1)", "full parameters" if full else "LoRA r=16",
                 "AdamW", "2e-5" if full else "1e-4", "600", "8 &times; 1,024",
                 "union (16,300)", "finetune_public.py"])
    rows = [
        open_row,
        ["&nbsp;&nbsp;Mamba pair &amp; Granite-4.0-H", "as above", "as above", "as above", "600",
         "2 &times; 768", "union", "finetune_public.py"],
        ["&nbsp;&nbsp;LoRA reference block (App. B)" if full else
         "&nbsp;&nbsp;full-parameter arms (App. B)", "LoRA r=16" if full else "full parameters",
         "AdamW", "1e-4" if full else "1e-4/2e-5/1e-5", "600", "8 &times; 1,024", "union",
         "finetune_public.py"],
        ["needle2", "vendor trainer", "vendor defaults", "&mdash;", "&mdash;", "&mdash;", "union",
         "vendor CLI"],
        ["This-work Table 1 rows", "full parameters", "AdamW", "2e-4, WSD", "1,500",
         "8 &times; 2,048 (accum 4)", "five-source", "localagent train sft"],
        ["This-work baserecipe (App. B)", "full parameters", "AdamW", "1e-4", "600",
         "8 &times; 2,048", "union", "localagent train sft"],
        ["Agentic arms (App. E)", "full parameters", "AdamW", "2e-4, WSD", "1,500",
         "8 &times; 2,048 (accum 4)", "five-source + episodes", "localagent train sft"],
        ["Open episode arms (App. E)", "full parameters", "AdamW", "2e-5", "600",
         "8 &times; 1,024", "union + episodes 12.5%", "finetune_public.py"],
    ]
    body = "".join(row_html(cells, numeric_from=99) for cells in rows)
    caption = ("Which configuration produced each row family. The open-block row follows the "
               "method gate; the Mamba and Granite-4.0-H exceptions are as stated above. "
               "Steps are optimisation steps; batch is micro-batch &times; "
               "max length. Open episode arms (SmolLM2-135M, Qwen3-0.6B): 2,329 episode "
               "decisions sampled at seed 2026 into the 16,300-row union (12.5%); one training "
               "seed per arm; control = the same model's Table 1 checkpoint; no episode row "
               "exceeds the 1,024-token window (longest 731). merged-v2 composition: ToolACE "
               "6,000, AndroidControl 6,000 and Mind2Web 2,694 public rows plus 1,606 "
               "author-synthetic multi-turn conversation rows (9.9%). Those rows serialize "
               "planning and tool chaining, but do not contain the simulator's goal/state/last-error "
               "decision observation. App. E's episode rows do; the two sources are not "
               "interchangeable. The generator ships with the supplementary code.")
    header = ("<thead><tr><th>Rows</th><th>Update type</th><th>Optimiser</th><th>Rate</th>"
              "<th class='n'>Steps</th><th>Batch</th><th>Corpus</th><th>Script</th></tr></thead>")
    return (f'<table data-caption="{caption}" class="long">' + header + "<tbody>"
            + body + "</tbody></table>")


def table_groups() -> str:
    """The three routes on one harness; every cell after fine-tuning, both metrics side by side."""
    no_exact = {"mcpatlas", "toolsandbox"}
    caption = ("Two-panel comparison under one harness: <em>function-name (exact) match</em>, %. "
               "Panel A compares released, generic and LocalAgent models at the shared 600-step "
               "stage; data and step budget match, while family rate/context differences remain "
               "(App. B). Only Panel A receives bold/underline maxima. Panel B reports the stronger "
               "shipped LocalAgent recipe and vendor-trained needle2 as unmatched deployment "
               "references; their accuracy is excluded from ranking. Columns "
               "separate dataset-author call splits from labelled projections/filtered views; "
               "these are not ten official scores. Throughput is architecture-level. Labels "
               "expose text-only, 204/17-row, first-action and name-only scope (App. A); MCP/TS "
               "omit exact match. Bold/underline mark per-column name/exact maxima. † needle2's "
               "self-reported 2-bit speed; ‡ unavailable cache.")
    display_suites = AUTHOR_CALL_SPLITS + STUDY_PROJECTIONS
    header = ("<thead><tr><th rowspan='2'>Model</th><th class='n' rowspan='2'>Params (M)</th>"
              "<th class='n' rowspan='2'>tok/s</th>"
              f"<th class='n' colspan='{len(AUTHOR_CALL_SPLITS)}'>dataset-author call splits</th>"
              f"<th class='n' colspan='{len(STUDY_PROJECTIONS)}'>study projections / filtered views</th>"
              "</tr><tr>"
              + "".join(f"<th class='n'>{T1_SUITE_TITLE[s]}</th>" for s in display_suites)
              + "</tr></thead>")

    entries = [ft_tag(tag) for tag, _ in PUBLIC]
    entries += [ft_tag(tag) for tag, _ in GENERIC]
    entries += [matched for _tag, _name, _pre, matched in OURS]
    best_name, best_exact = {}, {}
    for suite in SUITES:
        named = [(cell(t, suite, "type_match", T1_DIR), t) for t in entries]
        named = [pair for pair in named if pair[0] is not None]
        if named:
            best_name[suite] = max(named)[1]
        if suite not in no_exact:
            exact = [(cell(t, suite, "step_success_rate", T1_DIR), t) for t in entries]
            exact = [pair for pair in exact if pair[0] is not None]
            if exact:
                best_exact[suite] = max(exact)[1]

    def metric_cell(t: str, suite: str) -> str:
        name = cell(t, suite, "type_match", T1_DIR)
        name_text = f"<b>{number(name)}</b>" if best_name.get(suite) == t else number(name)
        if suite in no_exact:
            return name_text
        exact = cell(t, suite, "step_success_rate", T1_DIR)
        exact_text = f"<u>{number(exact)}</u>" if best_exact.get(suite) == t else number(exact)
        return f"{name_text} <span class='sub'>({exact_text})</span>"

    rows = [f"<tr class='group'><td colspan='{3 + len(SUITES)}'>"
            "Panel A — shared 600-step comparison: released instruction models</td></tr>"]
    open_rows = []
    for tag, name in PUBLIC:
        payload = report(ft_tag(tag), T1_DIR) or report(tag, T1_DIR)
        if payload is None:
            continue
        tokens = speed(tag)
        params = payload["parameters"] / 1e6
        no_speed = "—‡" if tag == "granite-350m" else "—"
        cells = [name, f"{params:.0f}", no_speed if tokens is None else f"{tokens:.0f}"]
        cells += [metric_cell(ft_tag(tag), suite) for suite in display_suites]
        open_rows.append((params, row_html(cells)))

    rows.extend(row for _, row in sorted(open_rows, key=lambda pair: pair[0]))

    rows.append(f"<tr class='group'><td colspan='{3 + len(SUITES)}'>"
                "Generic pretrained backbones under the same stage (§4)</td></tr>")
    generic_rows = []
    for tag, name in GENERIC:
        payload = report(ft_tag(tag), T1_DIR)
        if payload is None:
            continue
        tokens = speed(tag)
        params = payload["parameters"] / 1e6
        cells = [name, f"{params:.0f}", "—" if tokens is None else f"{tokens:.0f}"]
        cells += [metric_cell(ft_tag(tag), suite) for suite in display_suites]
        generic_rows.append((params, row_html(cells)))
    rows.extend(row for _, row in sorted(generic_rows, key=lambda pair: pair[0]))

    rows.append(f"<tr class='group'><td colspan='{3 + len(SUITES)}'>"
                "LocalAgent under matched data and step budget</td></tr>")
    for tag, name, _pre_tag, matched in OURS:
        payload = report(matched, T1_DIR)
        if payload is None:
            continue
        tokens = speed(tag)
        cells = [name + " (matched)", f"{payload['parameters'] / 1e6:.0f}",
                 "—" if tokens is None else f"{tokens:.0f}"]
        cells += [metric_cell(matched, suite) for suite in display_suites]
        rows.append(row_html(cells))

    rows.append(f"<tr class='group'><td colspan='{3 + len(SUITES)}'>"
                "Panel B — unmatched deployment recipes (reference only; excluded from maxima)"
                "</td></tr>")
    for tag, name, _pre_tag, _matched in OURS:
        payload = report(tag, T1_DIR)
        if payload is None:
            continue
        tokens = speed(tag)
        cells = [name + " (shipped)", f"{payload['parameters'] / 1e6:.0f}",
                 "—" if tokens is None else f"{tokens:.0f}"]
        cells += [metric_cell(tag, suite) for suite in display_suites]
        rows.append(row_html(cells))
    needle = report("needle2", T1_DIR)
    if needle is not None and report("needle2-tuned", T1_DIR) is not None:
        tokens = speed("needle2")
        cells = ["needle2 [37] (vendor)", f"{needle['parameters'] / 1e6:.0f}",
                 "—" if tokens is None else f"{tokens:.0f}†"]
        cells += [metric_cell("needle2-tuned", suite) for suite in display_suites]
        rows.append(row_html(cells))

    return (f'<table data-caption="{caption}" class="long models">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_generic() -> str:
    """The generic-backbone ladder of §3: released -> fine-tuned under the shared recipe."""
    caption = ("Generic pretrained backbones under the shared fine-tuning recipe (cells: "
               "fine-tuned (as released)); Table 1's generic block rescores the same backbones under "
               "the open block's matched stage on the full pool. Pythia [47] is Pile-pretrained at three scales; "
               "DistilGPT-2/GPT-2 [49] are WebText-lineage; the Mambas [48] are pure state-space models "
               "(LoRA on their in/x/dt projections; peft excludes out_proj and conv1d). "
               "Mamba fine-tuning used batch 2/1 at max length 768 to fit the torch-native "
               "scan's memory; steps, rank and learning rate unchanged.")
    ladder = [("pythia-70m", "Pythia-70M"), ("pythia-160m", "Pythia-160M"),
              ("pythia-410m", "Pythia-410M"), ("distilgpt2", "DistilGPT-2 (82M)"),
              ("gpt2", "GPT-2 (124M)"), ("mamba-370m-hf", "Mamba-370M"),
              ("mamba-790m-hf", "Mamba-790M")]
    suites = ("androidcontrol", "toolace", "xlam", "bfcl", "mobileactions")
    header = ("<thead><tr><th>Model</th><th class='n'>tok/s</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in suites)
              + "</tr></thead>")
    rows = []
    for slug, name in ladder:
        rel, ft = report(f"rel-{slug}"), report(f"ft-{slug}")
        if rel is None and ft is None:
            continue
        tokens = speed(f"rel-{slug}")
        cells = [name, "—" if tokens is None else f"{tokens:.0f}"]
        for suite in suites:
            before = cell(f"rel-{slug}", suite, "type_match")
            after = cell(f"ft-{slug}", suite, "type_match")
            cells.append(f"{number(after)} <span class='sub'>({number(before)})</span>")
        rows.append(row_html(cells))
    return (f'<table data-caption="{caption}" class="models">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_clean() -> str:
    """AndroidControl with the instruction-level duplicates removed."""
    arms = [("smollm2-135m", "SmolLM2-135M, as released"), ("lfm2-350m", "LFM2-350M, as released"),
            ("qwen25-05b", "Qwen2.5-0.5B, as released"),
            ("ft-smollm2-135m", "SmolLM2-135M, fine-tuned"),
            ("ft-lfm2-350m", "LFM2-350M, fine-tuned"),
            ("catalog-10m", "LocalAgent-11M"), ("catalog-16m", "LocalAgent-16M"),
            ("catalog-96m", "LocalAgent-96M")]
    rows, kept = [], None
    for tag, name in arms:
        payload = report(tag, "evalsuite-clean")
        if payload is None:
            continue
        kept = payload["rows_kept"]
        entry = payload["suites"]["androidcontrol_clean"]
        full_type = cell(tag, "androidcontrol", "type_match")
        full_exact = cell(tag, "androidcontrol", "step_success_rate")
        rows.append(row_html([name, number(entry["type_match"]), number(full_type),
                              number(entry["step_success_rate"]), number(full_exact)]))
    if not rows:
        return ""
    caption = (f"AndroidControl with instruction-level duplicates removed ({kept} of 200 rows "
               "survive). Dropping the screenshot collapses distinct screens into identical "
               "instruction strings, so a quarter of the test set has an exact twin in the "
               "training corpus. The clean subset is what survives of the original 200-row audit "
               "sample and the comparison column is the full pool, so the gap spans both changes: it "
               "costs the LocalAgent our models about 10–12 points of function-name match and 18–21 of "
               "exact match, and reverses the ranking against an as-released 0.5B model (61.7).")
    header = ("<thead><tr><th>Model</th><th class='n' colspan='2'>function-name match</th>"
              "<th class='n' colspan='2'>exact match</th></tr>"
              "<tr><th></th><th class='n'>clean</th><th class='n'>full pool</th>"
              "<th class='n'>clean</th><th class='n'>full pool</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_profile() -> str:
    """Does the open models' adaptation profile transfer as an optimizer prior?"""
    path = RESULTS / "analysis/lora_profile_summary.json"
    if not path.exists():
        return ""
    roles = json.loads(path.read_text())["summary"]["per_role"]
    arms = {"uniform": ["distill-10m", "unis101", "unis202"],
            "profile-weighted": ["profiled-10m"],
            "profile-inverted": ["profinv-10m", "invs101", "invs202"]}
    suites = ("androidcontrol", "agentnet", "toolace", "xlam")
    rows, present = [], False
    for name, tags in arms.items():
        values = {suite: [cell(tag, suite, "type_match") for tag in tags
                          if cell(tag, suite, "type_match") is not None] for suite in suites}
        if not any(values.values()):
            continue
        present = True
        cells = [f"{name} (n={len(values[suites[0]])})"]
        for suite in suites:
            got = values[suite]
            if not got:
                cells.append("—")
                continue
            mean = sum(got) / len(got)
            spread = (max(got) - min(got)) * 100
            cells.append(f"{number(mean)} <span class='sub'>±{spread / 2:.1f}</span>")
        rows.append(row_html(cells))
    if not present:
        return ""
    order = ", ".join(f"{role.replace('_', ' ')} {value * 1000:.1f}"
                      for role, value in sorted(roles.items(), key=lambda kv: -kv[1]))
    caption = ("Fine-tuning eight open families on the same corpus concentrates the update "
               "unevenly (mean ‖dW‖/‖W‖ per role, ×10³): " + order + ". Handing that profile to "
               "our model as per-module learning rates changes nothing it should: on "
               "AndroidControl, the one suite whose replicate range is under 2 points, both "
               "directions are worse than uniform, and the one apparent gain "
               "(AgentNet) sits inside a uniform range of 36 points. Function-name match (%), "
               "mean ± half-range over seeds.")
    header = ("<thead><tr><th>Per-module learning rate</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in suites) + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_wide() -> str:
    """Widening the training catalog distribution, holding the architecture and chain fixed."""
    arms = [("catalog-10m", "10.5M, union corpus (16,300 rows)"),
            ("wide-10m", "10.5M, union + xLAM train (34,300 rows)"),
            ("distill-10m", "10.5M, wide corpus, teacher-relabelled"),
            ("distill-96m", "95.3M, wide corpus, teacher-relabelled")]
    if any(report(tag) is None for tag, _ in arms):
        return ""
    caption = ("Two data-side interventions on our model, with the architecture, the chain and "
               "every hyper-parameter held fixed. Adding xLAM's training split buys xLAM (+27.5, "
               "2.6× its replicate spread) at the cost of device control; relabelling that corpus "
               "with the fine-tuned teacher gives the device-control points back and adds 4.5 on "
               "ToolACE, itself 1.5× that suite's spread. The model still sits below chance "
               "within the row's own candidate list on both novel-tool suites. Function-name "
               "match (%), exact match in parentheses.")
    rows = []
    for tag, name in arms:
        cells = [name]
        for suite in SUITES:
            match = cell(tag, suite, "type_match")
            exact = cell(tag, suite, "step_success_rate")
            cells.append(f"{number(match)} <span class='sub'>({number(exact)})</span>")
        rows.append(row_html(cells))
    delta = []
    for suite in SUITES:
        before = cell("catalog-10m", suite, "type_match")
        after = cell("distill-10m", suite, "type_match") or cell("wide-10m", suite, "type_match")
        delta.append("—" if before is None or after is None
                     else f"<b>{(after - before) * 100:+.1f}</b>")
    rows.append(row_html(["<b>net change vs union</b>"] + delta, css="floor"))
    odds = chance()
    if odds:
        rows.append(row_html(
            ["chance within the row's catalog"]
            + [number(odds[s]["random_within_catalog"]) if s in odds else "—" for s in SUITES],
            css="floor"))
    header = ("<thead><tr><th>Training corpus</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES) + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_donors() -> str:
    """Which open checkpoint is worth importing into a differently-shaped model."""
    donors = [("smollm2-135m", "SmolLM2-135M", 135), ("smollm2-360m", "SmolLM2-360M", 362),
              ("lfm2-350m", "LFM2-350M", 354), ("danube3-500m", "danube3-500M", 514),
              ("qwen25-05b", "Qwen2.5-0.5B", 494), ("qwen3-06b", "Qwen3-0.6B", 596)]
    caption = ("Importing weights from a differently-shaped open model into the same 10.5M "
               "model, then running the identical chain. <em>blocks</em> transfers attention and "
               "feed-forward matrices as leading sub-blocks rescaled to our model's variance, "
               "with donor layers mapped at even spacing; <em>+embed</em> additionally copies "
               "embedding rows for the tokens the two vocabularies share. Function-name match (%) "
               "on the two suites where naming is informative; the last rows are the same-family "
               "and random-initialisation controls. An em-dash marks a mode whose receipt was not produced.")
    naming_suites = ("toolace", "xlam")   # the caption's claim: naming-informative suites only
    rows, present = [], False
    for tag, name, params in donors:
        blocks, embed = report(f"dc-{tag}-blocks"), report(f"dc-{tag}-all")
        if blocks is None and embed is None:
            continue
        present = True
        cells = [name, str(params)]
        for payload in (blocks, embed):
            for suite in naming_suites:
                value = payload["suites"].get(suite, {}).get("type_match") if payload else None
                cells.append(number(value))
        rows.append(row_html(cells))
    for tag, name in (("rc-full", "same family, whole backbone"),
                      ("rc-scratch", "no transfer (random init)")):
        payload = report(tag)
        if payload is None:
            continue
        cells = [name, "10"]
        cells += [number(payload["suites"].get(suite, {}).get("type_match"))
                  for suite in naming_suites]
        cells += ["—", "—"]
        rows.append(row_html(cells, css="floor"))
    if not present:
        return ""
    header = ("<thead><tr><th>Donor</th><th class='n'>Donor (M)</th>"
              "<th class='n' colspan='2'>blocks</th>"
              "<th class='n' colspan='2'>blocks + embed</th></tr>"
              "<tr><th></th><th></th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in naming_suites) * 2
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def _region_asymmetry() -> str:
    """State the positional and structural asymmetry the region table shows, from its receipts.

    Section 4 asks where fine-tuning writes against where ability lives, and answers that the
    update map is a poor guide. That is a claim about update magnitude in one model. The region
    arms answer a different question - which parts of a pretrained sibling are worth adopting - and
    they are strongly asymmetric. Reporting only the first invites the reading that no part of the
    network is privileged, which these numbers do not support.
    """
    parts = {tag: _region_gain(tag, OPEN_CATALOG)
             for tag in ("early", "late", "attn", "ffn", "full")}
    return (
        "<p><b>What transfers is not evenly distributed.</b> Section 4 says the layers "
        "fine-tuning writes hardest are not the layers worth adapting, a statement about "
        "update magnitude within one model, not that no region is privileged; the "
        "same-shape transfer arms say the opposite. Averaged over all seven open-catalog suites "
        "against random initialisation, adopting the first half of the blocks gains %.1f points "
        "and the second half %.1f; attention gains %.1f and feed-forward %.1f, against %.1f for "
        "the whole backbone. Two of the four regions, the later blocks "
        "and the attention projections, carry nearly everything. Whether that is where the "
        "ability sits, or only where a sibling's weights happen to be reusable, these arms do not "
        "separate and we do not claim; they establish that the transferable content is "
        "concentrated, positionally and structurally rather than in proportion "
        "to parameter count.</p>"
        % (parts["early"], parts["late"], parts["attn"], parts["ffn"], parts["full"]))


def table_regions() -> str:
    order = ["scratch", "norms", "embed", "attn", "ffn", "early", "late", "no_embed", "full"]
    label = {"scratch": "none (random init)", "embed": "embedding table", "norms": "norm scales",
             "attn": "attention blocks", "ffn": "feed-forward blocks", "early": "first half",
             "late": "second half", "no_embed": "all but the embedding", "full": "whole backbone"}
    available = [(name, report(f"rc-{name}")) for name in order]
    available = [(name, payload) for name, payload in available if payload is not None]
    if not available:
        return ""
    # Parameter share per region, counted off our model's own checkpoint: the receipts never
    # carried the field, so the column the caption argues from used to render as em-dashes.
    shares_path = RESULTS / "analysis/region_shares.json"
    shares = json.loads(shares_path.read_text())["fractions"] if shares_path.exists() else {}
    caption = ("Same-family transfer: regions adopted from a pretrained sibling of identical shape, in place of the pretraining stage, so no projection is needed. "
               "Parameter share is close to inverted as a predictor of value. "
               "Function-name match (%).")
    rows = []
    best = max(range(len(available)),
               key=lambda i: available[i][1]["suites"].get("toolace", {}).get("type_match", 0))
    for index, (name, payload) in enumerate(available):
        values = [payload["suites"].get(suite, {}).get("type_match") for suite in OPEN_CATALOG]
        fraction = payload.get("region_fraction_copied")
        if fraction is None:
            fraction = shares.get(name)
        cells = [label.get(name, name), "—" if fraction is None else f"{fraction * 100:.1f}"]
        cells += [number(value) for value in values]
        rows.append(row_html(cells, bold={2, 3} if index == best else set()))
    header = ("<thead><tr><th>Region adopted</th><th class='n'>% of params</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in OPEN_CATALOG)
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")




def episode_report(name: str) -> dict | None:
    """Load one episode receipt from results/agentic, or None when it has not been produced."""
    path = RESULTS / "agentic" / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def table_episodes() -> str:
    """Episode success on the free stateful suite, against both model-free bounds.

    Scored on the state the policy actually reaches, so a well-formed call in the wrong order earns
    nothing. The step columns say what the policy was doing across the episodes it did not finish.
    """
    bounds = episode_report("free-eval-generated")
    rows: list[tuple[str, dict, dict | None]] = []
    if bounds:
        for key, label in (("oracle", "Oracle (replays the gold plan)"),
                           ("random", "Uniform over the nine device tools")):
            rows.append((label, bounds["policies"][key], None))
    for tag, label in (("11m-big", "LocalAgent-11M"), ("40m", "LocalAgent-43M"),
                       ("96m", "LocalAgent-96M")):
        before = episode_report(f"ep-fcall-{tag}-fixed")
        if before:
            rows.append((f"{label}, Table 1 checkpoint", before,
                         episode_report(f"ep-holdout-fcall-{tag}-fixed")))
    for arm, suffix in (("agentic arm", "agentic"), ("agentic arm, 4x corpus", "agenticxl")):
        for tag, label in (("11m", "LocalAgent-11M"), ("40m", "LocalAgent-43M"),
                           ("96m", "LocalAgent-96M")):
            after = episode_report(f"ep-{suffix}-{tag}-fixed")
            if after:
                rows.append((f"{label}, {arm}", after,
                             episode_report(f"ep-holdout-{suffix}-{tag}-fixed")))
    if not rows:
        return ""

    def pct(value: float | None) -> str:
        # An em-dash means the receipt does not carry the field, which is not the same as a zero:
        # the first baseline receipts predate the step profile.
        return "&mdash;" if value is None else f"{value * 100:.1f}"

    body = []
    for label, payload, holdout in rows:
        profile = payload.get("step_profile") or {}
        cells = [label,
                 pct(payload.get("episode_success")),
                 pct(payload.get("milestone_rate")),
                 pct(profile.get("schema_valid")),
                 pct(profile.get("transitioned")),
                 pct(profile.get("progressed")),
                 pct(None if holdout is None else holdout.get("episode_success"))]
        body.append(row_html(cells))
    caption = ("Episode success on the free stateful suite: 210 tasks over seven families, scored "
               "on whether the goal state holds when the episode ends rather than on whether each "
               "call matched a script. The oracle and the uniform policy bound the metric; the "
               "held-out column is a three-goal, twelve-step family that appears in no training "
               "corpus. Step columns are fractions of all steps taken, so they describe the "
               "episodes a policy did not finish as well as the ones it did. The uniform policy "
               "draws each tool's arguments from that tool's own recorded calls and is therefore "
               "schema-valid by construction; its column is not a capability.")
    names = ("Episode", "Milestone", "Schema ok", "State moved", "Progressed", "Held-out 3-goal")
    header = ("<thead><tr><th>Policy</th>"
              + "".join(f"<th class='n'>{name}</th>" for name in names)
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(body) + "</tbody></table>")


def _rename_prose() -> str:
    """Report the rename experiment from its receipts, including where it qualifies the claim."""
    arms = [("fcall-11m-big", "LocalAgent-11M"), ("fcall-96m", "LocalAgent-96M")]
    read = {}
    for tag, label in arms:
        payload = report("rename-" + tag, T1_DIR)
        if payload is None:
            continue
        suites = payload.get("suites", {})
        if not all(v in suites for v in ("xlam", "xlam_opaque", "xlam_shuffled")):
            continue
        read[label] = tuple(suites[v]["type_match"] * 100
                            for v in ("xlam", "xlam_opaque", "xlam_shuffled"))
    if len(read) < 2:
        return ""
    big = read["LocalAgent-96M"]
    small = read["LocalAgent-11M"]
    return (
        "<p><b>Is the xLAM gain keyed to names?</b> Rewriting only names and gold answers "
        "separates catalog reading from lookup. LocalAgent-96M is robust: "
        f"{big[0]:.1f} unmodified, {big[1]:.1f} with opaque identifiers and {big[2]:.1f} "
        f"with names rotated; LocalAgent-11M falls from {small[0]:.1f} to {small[1]:.1f}/"
        f"{small[2]:.1f}. Coverage predicts which suites move, but the 96M "
        "result survives name erasure and rotation; memorised lookup is the smaller rung's "
        "mechanism, not the imported inventory's only use.</p>")


def table_rename() -> str:
    """Does the xLAM result depend on the tool names themselves? Two rewrites of the same rows.

    The correlational evidence is that xLAM's evaluation names overlap the imported split almost
    completely while ToolACE's do not. That is consistent with the model having learned the
    catalog and with its having learned a name-to-function mapping. These two views separate them
    by rewriting only the names on the same 2,941 rows, leaving question, descriptions and
    parameter schemas untouched and rewriting the gold answer to match.
    """
    arms = [("fcall-11m-big", "LocalAgent-11M"), ("fcall-96m", "LocalAgent-96M")]
    body = []
    for tag, label in arms:
        payload = report(f"rename-{tag}", T1_DIR)
        if not payload:
            continue
        suites = payload.get("suites", {})
        if not all(view in suites for view in ("xlam", "xlam_opaque", "xlam_shuffled")):
            continue
        base = suites["xlam"]["type_match"] * 100
        cells = [label, f"{base:.1f}"]
        for view in ("xlam_opaque", "xlam_shuffled"):
            value = suites[view]["type_match"] * 100
            cells.append(f"{value:.1f} ({value - base:+.1f})")
        body.append(row_html(cells))
    if not body:
        return ""
    caption = ("Whether xLAM's result is keyed to the tool names. The same 2,941 rows with only "
               "the names rewritten: <em>opaque</em> replaces every name with tool_0, tool_1, &hellip; "
               "in catalog order, so a name carries no information and only the description and "
               "the parameter schema remain; <em>shuffled</em> rotates the names among the row's own "
               "tools, so each meaningful name sits on the wrong function and the gold answer is "
               "whichever name now sits on the right one. A model reading the catalog should be "
               "unmoved by shuffling; one going by the name is actively misled by it. Both are "
               "deterministic per row and rewrite the gold answer, so the task stays solvable. "
               "Function-name match (%), change from the unmodified suite in brackets.")
    header = ("<thead><tr><th>Model</th><th class='n'>xLAM</th>"
              "<th class='n'>Opaque names</th><th class='n'>Shuffled names</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(body) + "</tbody></table>")


def spread_96m() -> dict[str, float]:
    """Per-suite range over three seeds of the 96M Table 1 arm, measured this submission day.

    The band every ledger claim is read against comes from five runs of one 10.5M configuration.
    Whether it carries to other sizes was a stated caveat with nothing behind it; these receipts
    are the check. Empty when the seed receipts are absent, and every consumer degrades then.
    """
    tags = ("fcall-96m", "fcall-96m-s101", "fcall-96m-s202")
    out: dict[str, float] = {}
    for suite in SUITES:
        values = [cell(tag, suite, "type_match", T1_DIR) for tag in tags]
        if any(v is None for v in values):
            continue
        out[suite] = (max(values) - min(values)) * 100
    return out if len(out) == len(SUITES) else {}


def _band96_clause() -> str:
    """What the 96M seed check found, from the receipts; a plain concession when they are absent."""
    measured = spread_96m()
    if not measured:
        return "which it may not"
    exceed = [(suite, measured[suite], SPREAD[suite]) for suite in SUITES
              if suite in measured and measured[suite] > SPREAD.get(suite, float("inf"))]
    held = len(measured) - len(exceed)
    if not exceed:
        return ("checked at 96M with three seeds of the Table 1 arm, it holds on all "
                f"{held} suites")
    parts = ", ".join(f"{SUITE_TITLE.get(n, n)} ({m:.1f} against {b:.1f})" for n, m, b in exceed)
    clause = (f"checked at 96M with three seeds of the Table 1 arm, it holds on {held} of "
              f"{len(measured)} suites and is exceeded on {parts}")
    if any(n == "mcpatlas" for n, _, _ in exceed):
        clause += (", so the bolded MCP-Atlas gains of two to three points sit inside the seed "
                   "range measured at the very scale they are claimed at")
    return clause


def table_noise_sources() -> str:
    """Separate the two noise sources the replicate band conflates.

    The band prices training-run variance: five runs of one unchanged configuration, re-scored.
    It says nothing about how precisely a suite can measure a rate at all, which is set by its pool
    size. A suite whose sampling interval is wider than the band is limited by its pool, not by
    seeds, and the reverse for the others. Both are reported per suite so a reader can see which
    constraint binds where, rather than being handed one number to apply everywhere.
    """
    payload = report(OURS[2][0])
    if payload is None:
        return ""
    rows = []
    for suite in SUITES:
        entry = payload["suites"].get(suite)
        if not entry or not entry.get("rows"):
            continue
        n = entry["rows"]
        rate = entry.get("type_match")
        if rate is None:
            continue
        # Wilson 95% interval at this cell's own rate, the interval a bootstrap over rows
        # converges to and the one a proportion actually admits.
        z = 1.959963985
        half = (z / (1 + z * z / n)) * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5)
        sampling = half * 200            # full width, percentage points
        band = SPREAD.get(suite)
        if band is None:
            continue
        binds = "seeds" if band >= sampling else "pool"
        ninety_six = spread_96m().get(suite)
        rows.append(row_html([SUITE_TITLE.get(suite, suite), f"{n:,}", f"{sampling:.1f}",
                              f"{band:.1f}",
                              "&mdash;" if ninety_six is None else f"{ninety_six:.1f}",
                              binds], numeric_from=1))
    if not rows:
        return ""
    caption = ("Two noise sources, separately. <em>Sampling</em> is the 95% Wilson interval width for "
               "the LocalAgent-96M cell on that suite, the interval a bootstrap over rows converges "
               "to; it is what the pool size alone permits. <em>Replicates</em> is the band this "
               "paper reads claims against, the range over five runs of one unchanged 10.5M "
               "configuration; the 96M column re-measures it with three seeds of the 96M Table 1 arm, "
               "the check the first caveat below reports. The binding column names which of "
               "sampling and the 10.5M band is larger, and so which "
               "constrains a difference on that suite. The band is applied to sizes and "
               "interventions it was not measured on, which is the standing caveat; where the pool "
               "is the binding constraint instead, no number of seeds would help.")
    header = ("<thead><tr><th>Suite</th><th class='n'>Rows</th>"
              "<th class='n'>Sampling 95% (pts)</th><th class='n'>Replicates 10.5M (pts)</th>"
              "<th class='n'>Replicates 96M (pts)</th>"
              "<th class='n'>Binding</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def _mixer_where() -> str:
    """What the mixer axis actually trades, read across all three arms.

    A pooled mean is a poor summary here: it moves 27.9, 32.8, 33.0 across the axis and hides a
    systematic exchange underneath. The suites split into two families that move in opposite
    directions with attention count, so the sentence reports the split rather than the average.
    """
    arms = []
    for tag, attn_layers in (("mixer-attn-10m", 4), ("catalog-10m", 2),
                             ("mixer-convheavy-10m", 1)):
        payload = report(tag, T1_DIR)
        if payload is None:
            return ""
        arms.append((attn_layers, payload["suites"]))
    calling = ("toolace", "xlam", "bfcl", "toolbench")
    gui = ("agentnet", "androidcontrol", "mind2web")

    def family_mean(suites: dict, names) -> float:
        values = [suites[n]["type_match"] * 100 for n in names if n in suites]
        return sum(values) / len(values) if values else 0.0

    call_means = [family_mean(s, calling) for _, s in arms]
    gui_means = [family_mean(s, gui) for _, s in arms]
    biggest = max(
        ((n, (arms[2][1][n]["type_match"] - arms[0][1][n]["type_match"]) * 100)
         for n in gui if n in arms[0][1] and n in arms[2][1]),
        key=lambda kv: abs(kv[1]))
    return (
        " The mean is a poor summary of what happened. The suites split into two families that "
        "move in opposite directions with attention count. On the four function-calling suites the "
        "means run %.1f, %.1f, %.1f from four attention layers down to one, so attention helps; on "
        "the three GUI and desktop suites they run %.1f, %.1f, %.1f, so convolution helps, and the "
        "largest single movement is %s at %+.1f between the extremes. The hybrid sits between the "
        "two on both families rather than leading either, which is a more specific claim than the "
        "pooled mean can make and not the one this paper made before."
        % (call_means[0], call_means[1], call_means[2],
           gui_means[0], gui_means[1], gui_means[2],
           SUITE_TITLE.get(biggest[0], biggest[0]), biggest[1]))


def _mixer_episode() -> str:
    """The mixer arms on the episode suite, read from their receipts.

    The convheavy config's comment promised this reading: if attention count is what carries
    verbatim argument copying, the one-attention-layer arm should fall on it. The receipts answer
    a sharper question than the one asked.
    """
    rows = []
    for tag, layers in (("ep-mixer-attn-10m-fixed", "four"),
                        ("ep-mixer-hybrid-10m-fixed", "two"),
                        ("ep-mixer-convheavy-10m-fixed", "one")):
        payload = episode_report(tag)
        if payload is None:
            return ""
        rows.append((layers, payload["step_profile"]["schema_valid"],
                     payload["copy_fidelity"]["verbatim"], payload["episode_success"]))
    if any(r[3] > 0 for r in rows):
        return ""          # these are single-call checkpoints; a nonzero episode means a mixup
    schemas = ", ".join(f"{r[1]:.3f}" for r in reversed(rows))
    copies = ", ".join(f"{r[2]:.2f}" for r in reversed(rows))
    return (
        "<p><b>The mixer on episodes.</b> The convheavy config carried a prediction: "
        "if attention count carries verbatim copying, the one-attention-layer arm should "
        "fall on it. As episode policies all three complete nothing, as single-call "
        "checkpoints should; the step profile separates them. The fraction of steps whose "
        f"call satisfies its schema rises {schemas} from one attention layer to four at fixed "
        f"budget, while verbatim copy fidelity stays flat ({copies}). At this size attention "
        "carries assembling a legal call from a live state; the copying the "
        "config's guard names does not vary with it.</p>")


def _mixer_result() -> str:
    """Report the mixer ablation once its arms exist; say nothing while they do not."""
    arms = {}
    for tag, label in (("mixer-attn-10m", "attention-only"),
                       ("catalog-10m", "hybrid"),
                       ("mixer-convheavy-10m", "convolution-heavy")):
        payload = report(tag, T1_DIR)
        if payload is None:
            continue
        author = scope_mean(tag, AUTHOR_CALL_SPLITS, directory=T1_DIR)
        projected = scope_mean(tag, STUDY_PROJECTIONS, directory=T1_DIR)
        if author is not None and projected is not None:
            arms[label] = {"author": author, "projected": projected, "decode": speed(tag),
                           "params": payload.get("parameters")}
    if len(arms) < 2:
        return ""
    decodes = [a["decode"] for a in arms.values() if a["decode"] is not None]
    spread = (max(decodes) - min(decodes)) / max(decodes) * 100 if len(decodes) > 1 else None
    parts = ", ".join(
        "%s %.1f/%.1f" % (label, arm["author"], arm["projected"])
        for label, arm in arms.items())
    counts = "/".join(format(arm["params"], ",") for arm in arms.values() if arm.get("params"))
    sentence = ("<p><b>The mixer, priced.</b> Three arms hold chain, corpus, tokenizer, "
                "schedule and contract fixed, varying only how many of four blocks are attention "
                "(parameters %s, within %.2f%%). Dataset-author / study-projection descriptive "
                "means: %s.%s"
                % (counts, _param_spread(arms), parts, _mixer_where()))
    if spread is not None:
        fastest = max(decodes)
        slowest = min(decodes)
        sentence += (" On decode the mixer is worth something rather than nothing: %.0f "
                     "against %.0f tokens/s on four CPU threads, a %.1f%% difference &mdash; "
                     "not what puts either arm an order of magnitude ahead of a 0.5B open "
                     "model at 30 tokens/s; that is the parameter budget. The withdrawn "
                     "version reported 1.7%% here, which was two "
                     "checkpoints of one architecture and measured no mixer at all."
                     % (fastest, slowest, spread))
    return sentence + "</p>\n\n" + _mixer_episode()


def _param_spread(arms: dict) -> float:
    counts = [a["params"] for a in arms.values() if a.get("params")]
    if len(counts) < 2:
        return 0.0
    return (max(counts) - min(counts)) / min(counts) * 100


def table_mixer() -> str:
    """The mixer priced at fixed parameter count: four, two and one attention layer of four.

    Every row prints the parameter count its own receipt records. The claim this table replaces
    was wrong precisely because two rows shared a parameter count that no two different
    architectures could share, and printing the number makes that failure visible rather than
    something a reader has to reconstruct from run names.
    """
    arms = [("mixer-attn-10m", "attention only", "4 of 4"),
            ("catalog-10m", "hybrid (this work)", "2 of 4"),
            ("mixer-convheavy-10m", "convolution-heavy", "1 of 4")]
    body = []
    for tag, label, mix in arms:
        payload = report(tag, T1_DIR)
        if payload is None:
            continue
        author = scope_mean(tag, AUTHOR_CALL_SPLITS, directory=T1_DIR)
        projected = scope_mean(tag, STUDY_PROJECTIONS, directory=T1_DIR)
        if author is None or projected is None:
            continue
        params = payload.get("parameters")
        decode = speed(tag)
        body.append(row_html([label, mix,
                              "—" if not params else f"{params:,}",
                              "—" if decode is None else f"{decode:.0f}",
                              f"{author:.1f}", f"{projected:.1f}"]))
    if len(body) < 2:
        return ""
    caption = ("Pricing the mixer at fixed budget: the same chain, corpus, tokenizer, schedule and "
               "prompt contract, varying only how many of four blocks are attention. Parameter "
               "counts are printed from each arm's own receipt, because the claim this table "
               "replaces rested on two rows that shared a count no two different architectures "
               "could share. A zero-attention stack is not buildable: the model config requires at "
               "least one attention layer for verbatim argument copying. Decode is tokens/s on four "
               "CPU threads; the final columns are separate unweighted descriptive means for "
               "dataset-author call splits and study projections / filtered views (%).")
    header = ("<thead><tr><th>Mixer</th><th>Attention layers</th><th class='n'>Parameters</th>"
              "<th class='n'>Decode (tok/s)</th><th class='n'>Author splits</th>"
              "<th class='n'>Study projections</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(body) + "</tbody></table>")


def table_ftmethod() -> str:
    """Does the shared LoRA recipe handicap the open block? An adapter-vs-full-parameter check."""
    models = [("smollm2-135m", "SmolLM2-135M"), ("lfm2-350m", "LFM2-350M"), ("qwen3-06b", "Qwen3-0.6B")]
    arms = [("ft-{t}", "LoRA r=16, 1e-4 (earlier reference)"),
            ("ft-{t}-full", "full parameters, 1e-4"),
            ("ft-{t}-full-lr2e-5", "full parameters, 2e-5"),
            ("ft-{t}-full-lr1e-5", "full parameters, 1e-5")]
    rows = []
    for pattern, label in arms:
        cells = [label]
        for tag, _ in models:
            receipt = pattern.format(t=tag)
            author = scope_mean(receipt, AUTHOR_CALL_SPLITS)
            projected = scope_mean(receipt, STUDY_PROJECTIONS)
            cells.extend(["—" if author is None else f"{author:.1f}",
                          "—" if projected is None else f"{projected:.1f}"])
        if all(c == "—" for c in cells[1:]):
            continue
        rows.append(row_html(cells))
    if not rows:
        return ""
    caption = ("Is the open block held back by the earlier adapter recipe? The same corpus and the "
               "same 600-step budget, varying only the method and its learning rate, on three "
               "models spanning the block. At a learning rate chosen for it, full-parameter "
               "fine-tuning is within 2.7 points of the earlier LoRA recipe in both scope groups "
               "for the larger models, so the adapter is not "
               "what decides the comparison; 1e-4 is a normal LoRA rate and roughly five times a "
               "normal full-parameter one, which is why the full arm collapses there without ever "
               "losing its output format (parse rates stay at 99&ndash;100%). The exception is the "
               "smallest model, which prefers full parameters at the high rate in both groups. We "
               + ("score every open row with full parameters at 2e-5, which sums higher within "
                  "both scope groups over the block (the switch is a single gate, never per-model). "
                  if ft_method() == "full" else
                  "keep LoRA for every open row rather than pick a method per model. ")
               + "Each model reports dataset-author / study-projection descriptive means for "
               "function-name match (%), never one combined score.")
    header = ("<thead><tr><th rowspan='2'>Fine-tuning method</th>"
              + "".join(f"<th class='n' colspan='2'>{name}</th>" for _, name in models)
              + "</tr><tr>" + "".join("<th class='n'>author</th><th class='n'>projection</th>"
                                       for _ in models) + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_rate_asymmetry() -> str:
    """Price LocalAgent's 1e-4 matched arm at the open block's 2e-5 rate."""
    sizes = [("43M", "baserecipe-40m", "baserecipe2e5-40m"),
             ("96M", "baserecipe-96m", "baserecipe2e5-96m")]
    rows = []
    for label, base_tag, twin_tag in sizes:
        if report(base_tag, T1_DIR) is None or report(twin_tag, T1_DIR) is None:
            continue
        for tag, rate in ((base_tag, "1e-4"), (twin_tag, "2e-5")):
            author = scope_mean(tag, AUTHOR_CALL_SPLITS, directory=T1_DIR)
            projected = scope_mean(tag, STUDY_PROJECTIONS, directory=T1_DIR)
            rows.append(row_html([label, rate, f"{author:.1f}", f"{projected:.1f}"],
                                 numeric_from=2))
    if not rows:
        return ""
    caption = ("Learning-rate asymmetry in the matched LocalAgent control. Both rows use full "
               "parameters, the same union corpus, 600 steps and the same initial checkpoint; "
               "only AdamW's rate changes. Values are the two declared scope means (%), not a "
               "ten-suite leaderboard average.")
    header = ("<thead><tr><th>Model</th><th>Rate</th><th class='n'>Author splits</th>"
              "<th class='n'>Study projections</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_baserecipe() -> str:
    """Our own models put through the open baselines' fine-tuning recipe instead of their own."""
    sizes = [("11M", "fcall-11m-big", "baserecipe-11m", "mid-11m-big", "unionbudget-11m"),
             ("43M", "fcall-40m", "baserecipe-40m", "mid-40m", "unionbudget-40m"),
             ("96M", "fcall-96m", "baserecipe-96m", "mid-96m", "unionbudget-96m")]
    sizes = [row for row in sizes if report(row[2]) is not None]
    if not sizes:
        return ""

    caption = ("Our pretrained models under the open baselines' data and step budget instead of "
               "the LocalAgent shipped recipe. Pretraining, midtrain and prompt contract are identical to "
               "the Table 1 LocalAgent row; the SFT stage changes to the union corpus, 600 steps, "
               "batch 8 and AdamW at 1e-4. The no-SFT row is the shared initial checkpoint, and "
               "the union-at-our-budget row separates corpus from optimisation where available. "
               "Two asymmetries remain explicit: this trainer has no adapter path, so the matched "
               "arm is full-parameter, and maximum length stays 2,048 because at 1,024 the "
               "catalog-conditioned "
               "contract refuses to render at all (a row needs 1,042 tokens and a catalog is not "
               "truncatable), which is itself the asymmetry &mdash; the baselines fit in 1,024 only "
               "because their harness truncates the prompt. Function-name match (%).")
    header = ("<thead><tr><th>Fine-tuning recipe</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES)
              + "</tr></thead>")
    rows = []
    for label, own, matched, pre, samebudget in sizes:
        rows.append(f"<tr class='group'><td colspan='{1 + len(SUITES)}'>LocalAgent-{label}</td></tr>")
        # The pre-SFT checkpoint is the initialisation BOTH arms start from, so it is the reference
        # that says how much of the fall is "the recipe" and how much is simply "no SFT".
        pairs = [(own, "five-source recipe (Table 1)"), (matched, "the baselines' recipe")]
        if report(samebudget) is not None:
            # union corpus at OUR budget: separates the corpus from the optimisation
            pairs.append((samebudget, "the baselines' corpus at our budget"))
        if report(pre) is not None:
            pairs.append((pre, "no SFT (the shared initialisation)"))
        for tag, name in pairs:
            cells = [name] + [number(cell(tag, s, "type_match")) for s in SUITES]
            rows.append(row_html(cells))
        deltas = []
        for suite in SUITES:
            before, after = cell(own, suite, "type_match"), cell(matched, suite, "type_match")
            if before is None or after is None:
                deltas.append("—")
                continue
            change = (after - before) * 100
            readable = abs(change) > SPREAD[suite] and suite != "mind2web"
            deltas.append(f"<b>{change:+.1f}</b>" if readable else f"{change:+.1f}")
        rows.append(row_html(["change"] + deltas, css="floor"))
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_constrain() -> str:
    """Paired decode-mode table: free generation vs catalog-scored selection, per FC suite."""
    pairs = [("distill-10m", "11M, pre-import backbone"),
             ("fcall-11m-big", "LocalAgent-11M"), ("fcall-40m", "LocalAgent-43M"),
             ("wide3-96m", "96M, two-source SFT"), ("wide4-96m", "96M +Toucan"),
             ("fcall-96m", "LocalAgent-96M"), ("staged-fcall-96m", "96M, staged backbone")]
    fc_suites = [s for s in ("toolace", "xlam", "bfcl", "toolbench", "toolsandbox",
                             "mcpatlas", "mobileactions") if s in SUITES]
    pairs = [(tag, name) for tag, name in pairs if report(tag) is not None]
    if not pairs:
        return ""
    caption = ("Decode-time contract enforcement, paired per checkpoint where both modes have receipts: the same weights scored "
               "with free generation (free) and with the tool name chosen by length-normalised "
               "logprob over the row's own catalog (scored), function-name match (%). The "
               "constraint buys the chance-in-catalog floor on out-of-inventory suites and taxes "
               "in-inventory ones; on-miss keeps the free name when it is in-catalog and falls back "
               "to the scored choice otherwise. Neither mode is used for any other table in this report.")
    body = []
    for tag, name in pairs:
        free = [cell(tag, s, "type_match") for s in fc_suites]
        body.append(row_html([f"{name} (free)"] + [number(v) for v in free]))
        if report(f"{tag}-constrained"):
            scored = [cell(f"{tag}-constrained", s, "type_match") for s in fc_suites]
            body.append(row_html([f"{name} (scored)"] + [number(v) for v in scored]))
        if report(f"{tag}-fallback"):
            miss = [cell(f"{tag}-fallback", s, "type_match") for s in fc_suites]
            body.append(row_html([f"{name} (on-miss)"] + [number(v) for v in miss]))
    floors = _chance_floors()
    body.append(row_html(["<em>chance in catalog</em>"]
                         + [f"<em>{100 * floors[s]:.1f}</em>" if s in floors else "—"
                            for s in fc_suites]))
    header = ("<thead><tr><th>Checkpoint (decode mode)</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in fc_suites)
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(body) + "</tbody></table>")


def _chance_floors() -> dict[str, float]:
    path = RESULTS / T1_DIR / "chance-baseline.json"
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}
    remap = {"mobileactions": "mobileactions"}
    out = {}
    for key, row in payload.items():
        suite = "mobileactions" if key.startswith("mobileact") else key
        if isinstance(row, dict) and "random_within_catalog" in row:
            out[suite] = row["random_within_catalog"]
    return out


def table_seeds() -> str:
    arms = [("catalog-10m", "seed 2026 (ladder rung)"), ("rc-full", "seed 2026 (region arm)"),
            ("seed-101", "seed 101"), ("seed-202", "seed 202"), ("seed-303", "seed 303")]
    present = [(tag, name) for tag, name in arms if report(tag) is not None]
    if len(present) < 3:
        return ""
    caption = ("Replicates of one 10.5M configuration, differing only in training seed and "
               "run-to-run GPU nondeterminism. Only ToolACE is stable enough to separate single "
               "runs, which is why the transfer claims are read there. Function-name match (%).")
    rows, columns = [], {suite: [] for suite in SUITES}
    for tag, name in present:
        values = [cell(tag, suite, "type_match") for suite in SUITES]
        for suite, value in zip(SUITES, values):
            if value is not None:
                columns[suite].append(value * 100)
        rows.append(row_html([name] + [number(value) for value in values]))
    rows.append(row_html(["<b>spread (max − min)</b>"]
                         + [f"<b>{max(v) - min(v):.1f}</b>" if len(v) >= 3 else "—"
                            for v in (columns[s] for s in SUITES)]))
    header = ("<thead><tr><th>Replicate</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES) + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_contract() -> str:
    arms = [("localagent-union", "fixed action head (28.3M)"),
            ("localagent-scratch", "fixed head, no transfer (28.3M)"),
            ("catalog-10m", "catalog in the prompt (10.5M)")]
    caption = ("Why the custom architecture reads its action space from the prompt. A fixed action "
               "head can only emit tools seen in training, so on a held-out catalog the right "
               "answer is outside its output space; it still parses at 98%. Function-name "
               "match (%); the fixed-head arms are scored on their historical 200-row "
               "protocol; the regime is not replayable at full pools, and the contrast is "
               "qualitative.")
    # the fixed-head arms are a different regime (kind localagent) that the full-pool
    # resweep cannot replay; they keep their historical 200-row receipts, noted in the caption
    # only the suites that existed when the fixed-head arms ran; the newer suites are not
    # scoreable for a head that cannot emit unseen tools, so the columns are dropped, not dashed
    head_suites = [s for s in SUITES
                   if cell("localagent-union", s, "type_match", "evalsuite") is not None]
    rows = [row_html([contract] + [number(cell(tag, s, "type_match", "evalsuite"))
                                   for s in head_suites])
            for tag, contract in arms]
    floor = floors()
    if floor:
        rows.append(row_html(["majority-class floor"]
                             + [number(floor.get(s)) for s in head_suites], css="floor"))
    header = ("<thead><tr><th>Where the action space lives</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in head_suites)
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_corpus() -> str:
    manifest = json.loads((RESULTS / "dataset/merged-v2.manifest.json").read_text())
    teaches = {"synthetic": "serialized multi-turn/tool chaining; no simulator state",
               "toolace": "argument-level function calling",
               "mind2web": "web element selection",
               "androidcontrol": "mobile UI actions with coordinates"}
    caption = ("The union corpus, used identically for our pretraining chain and for every "
               "fine-tuned baseline. The split is disjoint at the level of rendered prompts rather "
               "than rows, so the many steps of one trajectory cannot straddle it.")
    rows = [row_html([source["source"], teaches.get(source["source"], "—"),
                      f"{source['train']:,}", f"{source['eval']:,}"], numeric_from=2)
            for source in manifest["sources"]]
    rows.append(row_html(["<b>total</b>", "", f"<b>{manifest['train']['rows']:,}</b>",
                          f"<b>{manifest['eval']['rows']:,}</b>"], numeric_from=2))
    return (f'<table data-caption="{caption}">'
            "<thead><tr><th>Source</th><th>What it teaches</th><th class='n'>Train</th>"
            "<th class='n'>Held out</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def table_prune() -> str:
    """Depth-pruning Qwen3-0.6B to 10 of 28 layers under four selection policies."""
    caption = ("Depth-pruning Qwen3-0.6B to 10 of 28 layers (~0.31B; the 151,936-entry tied "
               "embedding alone is 155M) under four layer-selection policies, each healed with "
               "the shared 600-step LoRA recipe (function-name match, %; as-pruned, every policy scores "
               "zero on every suite). Keeping the layers the update map moves <em>least</em> is "
               "the only policy retaining any function calling (the depth-side counterpart of "
               "the least-written adapter result), but no policy clears a chance-in-catalog "
               "floor on a function-calling suite: at this ratio and heal budget depth-pruning "
               "chooses how the model is broken, not whether.")
    policies = [("uniform", "uniform (every &asymp;3rd layer)"),
                ("first", "first 10 layers"),
                ("least_written", "10 least-written layers"),
                ("most_written", "10 most-written layers")]
    rows = []
    for tag, name in policies:
        cells = [name] + [number(cell(f"ft-qwen3prune-{tag}", suite, "type_match"))
                          for suite in SUITES]
        rows.append(row_html(cells))
    header = ("<thead><tr><th>Kept layers</th>"
              + "".join(f"<th class='n'>{SUITE_TITLE[s]}</th>" for s in SUITES)
              + "</tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(rows) + "</tbody></table>")


def table_suite_sources() -> str:
    """Reference and provenance for every evaluation suite: where each one comes from,
    which split is scored, and whether its train split ever enters a training corpus."""
    caption = ("Provenance and scope of the ten public evaluation suites. The first three "
               "dataset-author call splits and the seven study projections / filtered views "
               "are visually separated in Table 1; none is presented as an official leaderboard "
               "score. Rows = evaluation-pool size after "
               "normalisation; appendix tables score the full pool where their arms have "
               "full-pool receipts and the first-200 protocol otherwise. Suites marked "
               "<em>never</em> contribute no training rows; the three union sources (§3) and "
               "three §4 imports use the source's own train split, deduplicated against the "
               "evaluation pool at the rendered-prompt level.")
    suites = [
        ("AndroidControl [2]", "mobile GUI", "HF <span class='mono'>OfficerChul/Android-Control-84k</span> "
         "(text-only mirror)", "held-out, action-balanced; dedup audit keeps 343/904", "904",
         "union corpus (§3)"),
        ("Mind2Web [3]", "web GUI", "HF <span class='mono'>osunlp/Mind2Web</span>",
         "text-only view held out from pinned train file", "252", "union corpus (§3)"),
        ("AgentNet [5]", "desktop GUI", "HF <span class='mono'>xlangai/AgentNet</span> (OpenCUA)",
         "text-only view, evaluation-only", "12,482", "never"),
        ("ToolACE [1]", "function calling", "HF <span class='mono'>Team-ACE/ToolACE</span> "
         "@ <span class='mono'>6bda777c</span>", "held-out shard of the pinned release", "949",
         "union corpus (§3)"),
        ("xLAM [6]", "function calling", "HF <span class='mono'>Salesforce/xlam-function-calling-60k</span>",
         "dataset authors' test shard", "2,941", "wide import (§4)"),
        ("BFCL [38]", "function calling",
         "HF <span class='mono'>gorilla-llm/Berkeley-Function-Calling-Leaderboard</span>",
         "contract-compatible subset, 204/1,510", "204", "never"),
        ("ToolBench [17]", "function calling", "OpenBMB/ToolBench (ToolLLM) G1+G2+G3 DFS split",
         "first-action projection, evaluation-only", "687", "never"),
        ("ToolSandbox [39]", "function calling", "GitHub <span class='mono'>apple/ToolSandbox</span> scenarios",
         "single-tool, single-turn projection", "17", "never"),
        ("MCP-Atlas [40]", "tool use (MCP)", "HF <span class='mono'>ScaleAI/MCP-Atlas</span>",
         "first-action projection (495/500 open with a call)", "495", "never"),
        ("Mobile-Actions [41]", "mobile system tools", "HF <span class='mono'>google/mobile-actions</span>",
         "dataset authors' eval split", "961", "wide3 import (§4)"),
    ]
    rows = [row_html(list(suite), numeric_from=99) for suite in suites]
    return (f'<table data-caption="{caption}" class="long">'
            "<thead><tr><th>Suite</th><th>Domain</th><th>Source artifact</th><th>Split scored</th>"
            "<th class='n'>Rows</th><th>Train split used</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


# --------------------------------------------------------------------------- prose

ABSTRACT = """
Tool-calling research centres on large language models, leaving models in the tens of millions
of parameters underexamined. We give an end-to-end recipe for a 10 to 95M parameter caller,
spanning pretraining through episode supervision, and price its ingredients against nineteen
released checkpoints under one harness on ten public
evaluation suites, including explicitly labelled text-only, filtered and first-action projections.
Data and additional pretraining compute transfer selectively: an imported split improves predominantly the suites
whose tool inventory it shares, while longer pretraining improves selected suites with no imported
rows. Under the maps tested, coordinate-preserving projections from larger models do not
improve suite scores over random initialisation, though same-shape
weights transfer strongly. Single-call accuracy does not yield task completion in our
simulator: all single-call-trained checkpoints complete zero multi-step episodes. Episode
supervision at a small mixture share lifts completion from zero to double-digit rates
with no consistent single-call cost across the two scope groups, and an oracle decomposition identifies argument fidelity, rather than tool
ordering, as the larger remaining bottleneck in this simulator.
"""

INTRO = """
<p>Research on tool-calling agents advances on the back of large language models, and open
releases follow it: general families commonly bottom out near 0.6B [9,10], the smallest general
instruction baseline here is 135M [8], and the 45M needle2 [37] is a rare specialist. Yet many
device actions are short, typed calls rather than open-ended generation, making the 10M&ndash;95M
regime plausible and largely unmeasured. We study it stage by stage rather than report one final
checkpoint.</p>

<p>The practical constraint creates a second problem. A custom edge-sized shape forfeits compatible
released weights, while a catalog-conditioned caller must choose from tool descriptions supplied at
runtime rather than a fixed inventory compiled into its output layer. Pretraining must therefore
support catalog reading, fine-tuning must teach call syntax and arguments, and episode supervision
must turn isolated calls into state-conditioned behaviour. We ask what each stage contributes and
which knowledge can be imported as data or weights.</p>

<p>Our 10.5M/43M/95.3M family is the instrument. One contract, parser and chance-floor harness
scores nineteen released checkpoints on ten public evaluation suites, with projections labelled;
controlled comparisons use matched data and step budgets, while shipped recipes and needle2 are
separated as unmatched deployment references. An intervention ledger changes one recorded stage
at a time, and four-thread server-CPU decode prices the architecture independently of its recipe.</p>

<p>Three findings organise the paper. First, imported data transfers mainly to covered inventories,
whereas additional pretraining improves selected suites with no imported rows. Second, the tested
coordinate-preserving cross-shape projections fail although same-shape adoption transfers strongly
(§4). Third, evaluated single-call checkpoints complete no simulator episodes until sparse episode
supervision produces LocalAgent single-checkpoint estimates of 17.6&ndash;34.3%; separate one-seed
SmolLM2-135M and Qwen3-0.6B controls reach 8.6% and 54.3%. Every model
still fails the held-out composition (§5, App.&nbsp;E).</p>
"""

RELATED = """
<p><b>Small agents.</b> Toolformer [52], ReAct [53] and Gorilla [54] establish tool use; Belcak et
al. [14] argue that repetitive schema-bound calls suit smaller models, but study 1B&ndash;7B.
AgentCPM-GUI [4] specialises a larger model for visual device control. Shen et al. [15] and
Żywot et al. [16] recover weak callers by adding planner/caller specialists
or multi-agent orchestration at inference. Those systems trade orchestration for capability; we
hold inference to one policy decision and ask what remains two orders of magnitude lower, across
pretraining, SFT and episode supervision.</p>

<p><b>Edge callers.</b> TinyAgent [13] retrieves tools for 1.1B/7B models; MobileLLM [26] designs
sub-billion architectures; Octopus v2 [27] binds a fixed pool to tokens; Hammer [28] masks
irrelevant functions. Our models and needle2 [37] instead accept runtime descriptions, but
needle2 retrieves five and constrains decoding. Thus fixed-token systems buy latency by compiling
the inventory into weights, retrieval systems shorten the prompt, and our experiment isolates free
generation from a prompt-supplied, changing catalog. This distinction matters for MCP-style use:
adding a tool changes context for our contract, whereas fixed-token callers require vocabulary or
weight updates. Our throughput result concerns the former contract on one CPU, not a universal
mobile-runtime comparison.</p>

<p><b>Multi-step use.</b> BUTTON [55] and PARL-MT [56] train multi-turn composition and progress;
tau-bench [57] and ToolSandbox [39] evaluate stateful completion. We test whether sparse episode
supervision creates completion at 10M&ndash;95M, whether single-call accuracy transfers without it,
and what the added supervision costs on the original call suites. Unlike those broader interactive
environments, our simulator fixes nine tools and executable preconditions so that sequencing,
argument fidelity and state change can be separated; it is a controlled diagnostic, not a substitute
for their real-world scope.</p>

<p><b>Weight reuse.</b> Net2Net [18], bert2BERT [19], LiGO [20], HyperCloning [21] and Weight
Subcloning [22] transfer across sizes; Wiring Beats Blending [58] shows map structure matters.
The former methods preserve functions or learn mappings; tokenizer-aware maps such as WECHSEL
[25] address a related embedding mismatch. Our deliberately cheaper test asks
whether coordinate-preserving cross-family slices can replace pretraining, then uses same-shape
adoption as a positive control. We also test whether fine-tuning update magnitude predicts reusable
regions; Yu et al. [46] instead localise individual <em>super weights</em>. Our negative result is
therefore about the tested slices, not learned alignment, distillation or cross-shape transfer in
general (§4, App.&nbsp;B).</p>

{{figure:pipeline|Architecture and study map: a 16k-vocabulary decoder with 12 short-convolution
and six MQA blocks. Arrows locate the architecture and data/weight evaluation routes; details:
Apps. A and B.}}
"""

GROUPS = """
<p>We first fix the shared measurement contract and compare models; §4 prices data and weight
imports, §5 adds task completion, and Apps. A and E give the full projection and episode protocols.</p>

<p><b>The evaluation pipeline (Q4).</b> Three dataset-author call splits and seven labelled
projections/filtered views flow through one prompt contract, parser and scorer (App. A). Catalogs
contain 2.1&ndash;33 candidates, so we report majority and chance-in-catalog floors [29] and gate
differences by measured replicate spread. Panel A uses the same 16,300-row union, sampled rows,
600-step budget and update type; rates and contexts remain family-specific (App. B). Function-name
match scores the action, exact match adds arguments; throughput is 64-token decode after a
512-token prefill on four CPU threads.</p>

<p><b>Models under test.</b> LocalAgent-11M/43M/96M (10.5M, 42.8M, 95.3M parameters) are one decoder family with a 16k vocabulary. Table&nbsp;1A gives their matched-data/step-budget SFT rows; Panel&nbsp;B gives the stronger five-source shipped SFT after the same 16,000-step pretraining and 700-step midtrain. Every accuracy cell is measured here; needle2's throughput is the exception.</p>

{{table:groups}}

<p><b>What the controlled panel establishes.</b> Within the eleven-model released-instruction block,
fine-tuning improves @FTUP@ of @FTTOT@ model-suite cells; the separate generic-backbone block is
not included in that count. Within Panel&nbsp;A, @STRONGEST@ has the best function-name match on @NBEST@; @OTHERWINS@. ToolBench makes the gap clear: chance is 22.9%, every instruction-tuned release clears it (@TBLO@&ndash;@TBHI@%), while matched LocalAgent-96M reaches 23.6%. Mind2Web remains majority-floor dominated (@M2WLO@&ndash;@M2WHI@; ours 80.2). We report no pooled rank: matched LocalAgent-96M scores 39.3/45.6 on the author/projection scope means, near Pythia-160M's 38.2/36.4 and below DistilGPT-2's 53.4/47.8. Panel&nbsp;B's shipped 96M reaches 76.8/43.5, but that gain is recipe evidence, not an architecture comparison; needle2 is likewise external. The robust architecture result is throughput: 604 tok/s for 11M versus about 30 for 0.5B models on four server-CPU threads; needle2's self-reported 2-bit speed is incomparable.</p>

"""

GROUPS = _fill(GROUPS, _who_leads_numbers())

LEDGER = """
<p>With the baseline tradeoffs established, each intervention in Table 2 runs against the control isolating it, chain and hyper-parameters fixed. Terms: <em>adoption</em> copies weights between identical shapes, <em>projection</em> crosses a shape change; <em>union</em> arms train on the union corpus, <em>wide</em> adds xLAM's split, <em>+teacher</em> relabels it.</p>

<p><b>Generic backbones as a control for purpose pretraining.</b> Under the matched stage, Pythia's
author/projection means scale @GP70AUTHOR@/@GP70PROJECTED@ (70M),
@GP160AUTHOR@/@GP160PROJECTED@ (160M), @GP410AUTHOR@/@GP410PROJECTED@ (410M);
LocalAgent-96M reaches @GLOCALAUTHOR@/@GLOCALPROJECTED@ and DistilGPT-2-82M
@GDISTILAUTHOR@/@GDISTILPROJECTED@. Only ToolBench separates LocalAgent from Pythia-160M
beyond replicate spread (@GTBDELTA@), with LocalAgent still near chance. Thus the matched 96M
sits near Pythia-160M and below Pythia-410M, not at a universal purpose-pretraining advantage;
corpus, tokenizer, architecture and budget still co-vary.</p>

<p><b>Covered-inventory data transfers; tested cross-shape slices do not.</b> Importing xLAM's
18,000-row split moves xLAM +26.8, 3.4&times; its spread, from below to above chance. Teacher
relabelling recovers device-control points only within spread. By contrast, no tested
coordinate-preserving projection clears an open-catalog spread; learned mappings such as LiGO
[20] remain untested.</p>

<p><b>Inventory coverage dominates the imported-data effect.</b> The split covers 99.4% of xLAM
evaluation names but 1.0% of ToolACE and 0% of ToolBench gold names; only xLAM moves. Renaming
costs the 96M rung under four points but the 11M rung thirty-two (App. B), making name lookup a
size-dependent shortcut [17,28]. Longer pretraining moves ToolACE +12.1 and ToolBench +9.0
beyond replicate spread; BFCL's +11.7 remains within its wide spread and only 1.8 points above
its chance floor, so chance clearance is not established there. A fresh-corpus control reproduces
ToolACE but gives back 8.1 points on ToolBench, identifying the latter gain as repetition-keyed
(App. C). ToolACE's sequential 39.2&rarr;51.3&rarr;59.3 path does not establish
additivity: coverage-associated gain is +4.0 at 1,599 steps and +8.0 at 16,000, but the arms are
not a matched 2&times;2 factorial. Thus ToolACE is the corpus-robust evidence that pretraining can
improve a suite with no imported rows; BFCL and ToolBench do not independently establish that
mechanism (App. A).</p>

{{table:ledger}}

<p><b>Same-shape weights remain reusable.</b> Against random initialisation, the whole backbone
gains @RGFULLAUTHOR@/@RGFULLPROJECTED@ on author/projection means; attention alone retains
@RGATTNAUTHORSHARE@%/@RGATTNPROJECTEDSHARE@%, while feed-forward gains only
@RGFFNAUTHOR@/@RGFFNPROJECTED@. Yet fine-tuning writes hardest in feed-forward projections, and
the least-written layer third recovers 85% of all-layer gain against 65% for the most-written
third. Update magnitude therefore does not predict useful adoption or adaptation here. Every
cross-shape donor (1.4&times;&ndash;6.3&times;) keeps device-control accuracy but loses open-catalog
performance; full anatomy and controls are in App. B.</p>

"""

LEDGER = _fill(LEDGER, _generic_ladder_numbers())
LEDGER = _fill(LEDGER, _region_gain_numbers())

LAYERS = """
<p><b>Which layers, and where fine-tuning writes</b> (full anatomy, App. B). Where shapes match, no region suffices and two dissociate sharply: attention alone retains most of what the whole backbone gives (21.8/27.2 vs 32.3/36.8 on ToolACE/xLAM), the embedding about as much despite holding 59.8% of the weights, the feed-forward blocks almost nothing (0.6/2.0).
Yet the role profile (mean update per projection type, eight families) shows fine-tuning writing hardest exactly there, in the feed-forward up and gate projections. Parameter share is nearly inverted as a predictor of value: a vocabulary change, forcing a fresh embedding, costs far less than its parameter count suggests. Per-layer updates add a second regularity across eight of the eleven families, Figure 2: below the model's own mean through the first third, above it thereafter, surviving the change from GQA transformer to convolution hybrid to mamba hybrid, and mild enough that depth alone is a weak prior.</p>

(The per-layer update profile, below the model's own mean through the first third and above it thereafter across eight of eleven families and three architecture classes, is figure 2 of the repository build.)

<p><b>The least-written third recovers most of the gain.</b> If ability lived where the update concentrates, adapting the most-written third should recover the all-layer recipe. The opposite holds: over three seeds per band on two models, the least-written third recovers 85% of the gain against the most-written third's 65%, beating it by 13 to 16 points on AndroidControl with no seed-range overlap; a random third lands between. Depth-pruning Qwen3-0.6B to 10 of 28 layers agrees: keeping the least-written layers is the only policy of four retaining any function calling (ToolACE 10.0, xLAM 16.2, BFCL 8.8 against 0.0 to 3.4), though none clears a chance floor.</p>

<p><b>What projection fails to replace is pretraining, specifically.</b> A larger donor is cut harder: every donor from 1.4&times; to 6.3&times; lands at the floor on every open-catalog suite, including the same-family, same-tokenizer arm where only the shape differs; the slice across a shape change destroys the information. The failure is localised: donor arms keep full device-control accuracy and score zero exactly where the pretrained control clears its floor. The mobile action space is learnable from the SFT corpus alone; catalog reading is what pretraining buys.</p>

<p><b>Two further routes, both rejected</b> (App. B and C): initialising the same pretraining from a projected checkpoint is mostly seed noise, gated by corpus <em>content</em> rather than freshness (the skeleton's edge is large over distillation traces, absent over equally-fresh generic chat, restored over open agentic corpora [42&ndash;44]); and transferring the <em>shape of the adaptation</em> clears nothing either way.</p>
"""

STRESS_LIMITS = _rename_prose() + """

<p><b>General-suite cost under the earlier LoRA reference is small on average, not uniformly.</b>
On HellaSwag, Winogrande, ARC-e/c, OpenBookQA and MMLU, the eleven released baselines' mean change spans &minus;1.8 to
+0.7 points, inside the &plusmn;2.9-point sampling error; Granite-4.0 nevertheless loses
5.7&ndash;7.3 on ARC-e. These receipts use the old LoRA stage, not Table&nbsp;1's current
full-parameter stage; the custom family was not run here, so no preservation claim follows.</p>

<p><b>Evidence hierarchy and practical reading.</b> Table&nbsp;2's within-family pairs hold the
recorded chain fixed, but most are single checkpoints and cross-family placement is descriptive.
Bold cells exceed a five-run range measured at 10.5M; pool uncertainty and multiplicity are in
App. A. Episode task tests compare fixed checkpoints, three-seed ranges compare LocalAgent
procedures, and the one-seed open arms establish no scale ordering (App. E). Operationally,
covered SFT buys represented inventory, additional pretraining robustly lifts ToolACE without
imported rows while the ToolBench gain is repetition-keyed, and same-shape but not tested
cross-shape weights remain useful.
Stateful execution requires episode decisions and remains bounded by trained task schemas.</p>

{{table:general}}

<p><b>Claim boundaries.</b> First, LocalAgent's deployment result is four-thread server-CPU decode;
App.&nbsp;A's Galaxy S22 and Tab A8 measurements are for needle2's 2-bit vendor engine and are
not directly comparable, so LocalAgent phone latency, memory, energy and thermals remain unmeasured.
ToolBench is first action, ToolSandbox has 17 rows, MCP-Atlas lacks schemas, BFCL retains 204 of
1,510 candidates, AndroidControl loses 62.1% under text-view deduplication, and the text-only GUI
suites are floor-dominated. Table&nbsp;1 therefore reports dataset-derived suite scores under our
harness, not ten official benchmark scores. Most interventions are one
checkpoint; the band is one configuration's threshold, not a per-model interval, and suite means
are unweighted. Cross-family rows co-vary corpus, tokenizer, architecture, context, rate and budget;
the shipped recipe is unmatched, and literal audits exclude direct but not semantic or upstream
overlap. Task completion exists only in our simulator (§5, App.&nbsp;E), with every model at zero on the
fully held-out composition.</p>
"""


def _loss_curve_appendix() -> str:
    """Describe the matched loss traces without claiming that LM loss is downstream transfer."""
    path = RESULTS / "analysis/losscurves_perstep.json"
    curves = json.loads(path.read_text())
    labels = {
        "pre-rand-96m": "random",
        "pre-skel-96m": "skeleton",
        "pre-anti-96m": "feed-forward",
        "pre-embed-96m": "embedding-only",
    }
    missing = [tag for tag in labels if tag not in curves or not curves[tag].get("val")]
    if missing:
        raise RuntimeError("missing loss-curve receipts: " + ", ".join(missing))

    initial = {tag: curves[tag]["val"][0][1] for tag in labels}
    final = {tag: curves[tag]["val"][-1][1] for tag in labels}
    random_final = final["pre-rand-96m"]
    projected = [tag for tag in labels if tag != "pre-rand-96m"]
    endpoint_text = ", ".join(f"{labels[tag]} {final[tag]:.3f}" for tag in labels)
    gains = [random_final - final[tag] for tag in projected]
    start_low, start_high = min(initial.values()), max(initial.values())
    gain_low, gain_high = min(gains), max(gains)
    return f"""
<p><b>Matched optimisation traces after cross-shape initialisation.</b> Figure&nbsp;3 reads the
actual 16,000-step matched runs. Initial validation loss spans
{start_low:.3f}&ndash;{start_high:.3f}; endpoints are {endpoint_text}. Projection finishes
{gain_low:.3f}&ndash;{gain_high:.3f} below random, so optimisation does not fail, but no arm clears
an open-catalog replicate spread (Table&nbsp;2): lower language-model loss is not behavioural
transfer. Same-shape adoption bypasses pretraining and is tested by the region controls.</p>

{{{{figure:losses|Matched 96M pretraining after random or cross-shape projected
initialisation: smoothed per-step training loss; markers show validation every 2,000 steps.}}}}
"""


SUPP_TRANSFER = """
<p><b>How each model family is fine-tuned.</b> @OPENFTDESC@ The objective is token-level cross-entropy with prompt tokens masked (only the answer span trains) under each model's own chat template. Declared exceptions: the Mamba baselines take the same full-parameter updates at batch 2, maximum
length 768, for the torch-native scan's memory, steps and rate unchanged (the released
ladder in App. C keeps their earlier in/x/dt adapter exception); needle2
uses the vendor's trainer &mdash; the one engine-boundary crossing &mdash; its confidence head not updated (App. A). Our models take full-parameter SFT: length 2,048 under the openai-full-catalog contract, AdamW 2e-4 with WSD, micro-batch 8, accumulation 4 (effective batch 32), 900 steps on two-source recipes, 1,500 on five-source (step count saturated; App. C). The
weight-transfer arms below inherit these settings, varying only initialisation.</p>

{{table:recipes}}

<p><b>Fairness of the shared recipe to the open models.</b> @FAIRNESSLEAD@ Re-fine-tuning three block-spanning models with full parameters at three learning rates on the same corpus and budget lands the two methods within two points on both larger models at a rate chosen for full updates; the adapter is not carrying the comparison, and one method serves every open row.</p>

{{table:ftmethod}}

<p><b>Fine-tuning matched, pretraining each party's own.</b> The matched control gives our base
the open block's union corpus, 600 steps, batch 8 and AdamW 1e-4.@BASERECIPENOTE@@RATE_TWIN@
We report no combined rank: LocalAgent-96M scores 39.3/45.6 on the dataset-author / study-projection
groups while needle2 scores 74.0/30.7, so the apparent tie under the former pooled mean was an
artefact of mixing scopes. Per-suite values and all three sizes are in the table below. This is
generous to us: midtrain already consumed the union corpus, and
@UNMATCHED@ our 2,048-token catalog is untruncated where open harnesses use 1,024.</p>

{{table:rate_asymmetry}}

{{table:baserecipe}}

<p><b>Where the five-source points sit.</b> Every readable 43M/96M difference is on a
function-calling suite whose split the five-source stage adds; uncovered suites do not move.
The apparent 96M device gains are inside spread (AndroidControl +11.8 versus 14.8;
AgentNet +21.5 versus 25.3, with parse rate 67.6&rarr;98.6), so we claim none. Mobile-Actions
shows naming rather than formatting: the matched arm parses 93.4% versus 82.5% yet names the
tool 17.5% versus 81.9%. On the three largest drops, union/five-source name coverage is
14.3/100, 1.9/95.0 and 0.0/70.1%; corpus replacement buys +9.5/+16.4/+12.4 while extra
budget buys only &minus;2.6 to +2.5. ToolACE's &minus;14.9 at nearly fixed coverage likely
reflects teacher relabelling; AgentNet remains floor-dominated.</p>



<p><b>The least-written third recovers most of the gain.</b> Placement is documented as forgiving [30&ndash;32], shift-dependent [33], a poor guide to intervention [34, 35]; the band test: if ability lived where the update concentrates, the top third should recover the all-layer recipe and the bottom third should not. The opposite holds: over three seeds per band on two models, the bottom band recovers 85% of the gain against the top's 65%, beating it by 13&ndash;16 points on AndroidControl with no seed-range overlap; a random third lands between. Update magnitude records where optimisation wrote, not where ability lives; scale relocates the site without changing it (Figure 2b). Depth pruning agrees: cut to 10 of 28 layers, Qwen3-0.6B keeps any function calling only under the <em>least-written</em> policy of four (ToolACE 10.0, xLAM 16.2, BFCL 8.8 against 0.0&ndash;3.4), though none clears chance: depth alone only chooses how the model breaks.</p>

{{figure:layers|Each layer's ‖ΔW‖/‖W‖ divided by its model's mean. (a) Mean and min–max band
over nine open families across three architecture classes: below the model's own mean through
the first third and above it thereafter for seven families; two SmolLM2 variants are exceptions.
(b) Qwen3 at three scales: the band moves later and turns bimodal at 4B.}}

<p>When shapes match and no projection is needed, parameter share is nearly inverted as a
predictor of value: the embedding table, the largest region in a 16k-vocabulary model at this
scale (59.8% of all weights), is worth less alone than the 40.2% that is everything else. The
computation in the blocks transfers, not the lookup table in front of them: a vocabulary change,
which forces a fresh embedding, costs far less than its parameter count suggests.</p>

""" + _loss_curve_appendix() + """

<p><b>What the cross-shape result does and does not cover.</b> The negative result is specific to
coordinate-preserving projections &mdash; leading-block truncation, head-slicing and this
appendix's size-ratio width crops &mdash; which carry nothing across an architecture boundary that
random initialisation does not already give on the downstream suites. Untested methods that could each change the answer:
a learned projection or growth operator in the LiGO style [20], hidden-state or feature
distillation, logit distillation, alignment of layers before mapping, low-rank factorisation of
the donor, function-preserving reductions, structured pruning with re-training, and
tokenizer-aware embedding conversion; only the last two have partial evidence, from the
depth-pruning arms above. With the same-shape arms transferring strongly, the supported
statement is that naive
cross-shape slicing fails, not that cross-shape transfer is impossible.</p>
""" + _region_asymmetry()
SUPP_TRANSFER = SUPP_TRANSFER.replace("@OPENFTDESC@", _open_ft_description())
SUPP_TRANSFER = _fill(SUPP_TRANSFER, _method_tokens())
SUPP_TRANSFER = SUPP_TRANSFER.replace("@RATE_TWIN@", _rate_twin())

SUPP_SETUP = """
<p><em>These appendices expand the main text's analyses; every headline number is
in the body.</em></p>

<p>Each suite's source artifact, scored split, training-corpus use, and reference:</p>

{{table:suite_sources}}

<p><b>Asset licences.</b> ToolSandbox is used under the Apple Sample Code License; MCP-Atlas is
CC-BY-4.0. The pretraining-source table below records the remaining corpus licences.</p>

<p><b>The AndroidControl deduplication.</b> The text-only mirror replaces screenshots with a
fixed prefix plus the instruction, so episodes differing only in screen state collapse into
byte-identical user turns; a test row is dropped when its user turn occurs in the training
split. The 200-row protocol lost a quarter of the sample; the full 904-row pool loses
561 rows (62.1%), contamination larger than the sample suggested. Among surviving twins, one
shares its gold call verbatim across splits (pure memorisation credit) while another's golds
differ only in capitalisation, charging the memoriser an exact-match error &mdash; why exact
match falls harder than name match once twins are removed.</p>

<p><b>ToolSandbox, what the projection shows.</b> Our catalog-conditioned models emit fully parseable calls yet select the gold tool 0 of 17 times, indistinguishable from the 3.0% chance-in-catalog floor. Ranking all 33 candidates by length-normalised sequence logprob at an 8,192-token window hides no weak preference: mean gold rank 18.4/18.9 against a uniform 17.0 (z&nbsp;=&nbsp;+0.6/+0.8), reciprocal rank 0.114/0.099 against 0.124, top-five 2 of 17 against 2.6 expected. A disclosed control closes the loop: adding 1,066 conversations synthesised from the same 33 public schemas (no test scenario read) still selects gold 0 of 17, every other suite inside its spread. The deficit is not name coverage.</p>

<p><b>ToolBench's floor</b> is the most informative of the function-calling suites: 5.9
candidates per task against 2.1–3.4 for the others puts chance within the row's own catalog
at 22.9% rather than the 45.8–61.4% those suites leave &mdash; the most headroom and the
sharpest test of catalog reading; the
novel-inventory suites leave lower floors still (MCP-Atlas 6.7%, ToolSandbox 3.0%).</p>

<p><b>needle2, and what is and is not claimed.</b> Cactus needle2 [37], the one released
model in our size class (a 45M tool-calling specialist), is a 2-bit engine with
grammar-constrained decoding, a top-5 tool-retrieval head and a confidence
gate that abstains and escalates; scored through its own engine, catalogs above
five tools pass through the retrieval head and abstentions count as unparsed, folding coverage
and correctness together. As precision-when-answering (200-row audit):
90–100% on the four function-calling suites at 27–37% coverage; its one lead,
ToolSandbox (64.7 vs our pretrained 0/17), may reflect better compatibility with its pretrained
device-assistant distribution or retrieval; this study does not isolate the cause. Its tok/s cell is the
engine's self-reported decode rate beside f16 models under the 512-prefill/64-decode protocol
&mdash; not apples-to-apples &mdash; and its android build brackets a 5.8× hardware
range: a Galaxy S22 sustains 643–654 tok/s at 23 MB peak RAM (top of the maker's 300–700
claim), a sub-$200 Galaxy Tab A8 112 tok/s, below the claimed floor though
interactive &mdash; this study's only on-device measurements.</p>

<p><b>needle2, fine-tuned.</b> The tuned variant trains through the vendor's own trainer,
crossing the engine boundary; it gains open-catalog calling like every
model under the recipe (xLAM 34.3&rarr;89.4 at full pools) but, alone among them, loses
device control (AndroidControl 37.9&rarr;23.3, where all eleven open baselines gain, +17.4 to
+78.0). The vendor's trainer does not update the confidence head: the tuned row
reports no calibrated confidence and does not share the abstention behaviour
above.</p>

<p>The pretraining streams and the midtrain and SFT conversation splits are tabulated
below; one SFT harness consumes them identically for
every model, so a row in Table 1 or 2 differs from its neighbour only in the splits its
recipe names.</p>

{{table:pretrain_sources}}

{{table:train_sources}}

<p><b>One row set for every model.</b> BFCL writes Python-flavoured schema types, and its gold arguments do not always satisfy their own declared schema; a model that builds its own catalog never notices, while a catalog-conditioned model's contract refuses such rows and emits nothing, which the scorer counts wrong &mdash; under the then-200-row protocol our models saw 94 of the 200 sampled rows and the others saw all of them. The normaliser now coerces the schemas and keeps only rows the contract will render (204 of 1,510 candidates survive), so both groups are scored on identical data; the correction moved LocalAgent-96M from 27.5% to 51.5% on BFCL and its parse rate from 43.5% to 91.2%. The filter applies identically to every model; like ToolBench, this subset is not an official BFCL score and should not be quoted as one.</p>

<p><b>The replicate noise floor.</b> The per-suite spread every claim is read against is the
range across five runs of one unchanged 10.5M configuration, four training seeds between
them, re-scored on the full pools; each suite's spread is additionally floored at the value
of one row of its pool: a 17-row suite cannot resolve a finer difference whatever the
replicates do. The band is Table 2's last row.</p>

<p><b>Whether that floor does statistical work, and where it does not.</b> It is a range over
replicates, not a confidence interval, and Table 2 reads its 21 interventions
against every readable suite cell, so multiplicity is a fair concern. Under a two-proportion z
test on each cell's own pool sizes, the band is <em>stricter</em> than a
Bonferroni-corrected test on most cells but not on all. {{MULTIPLICITY}} Three caveats stand:
the band is measured on one 10.5M configuration and assumed to carry &mdash; {{BAND96}}; it
prices training-run variance while the z test prices only sampling; and on a small pool the
band can be narrower than the interval that pool admits &mdash; the exceptions above.
Where they disagree we obey the band, except those cells. The z test
is also unpaired where McNemar would have more power; the weaker test is
conservative.</p>

<p><b>Which noise source binds, per suite.</b> The band prices training-run variance, not
how precisely a suite can measure a rate &mdash; its pool size fixes that; below, the
95% Wilson interval for a cell on each suite is set against the replicate band. On the large
pools the band is the
larger and seeds bind; on the small ones the pool binds, and no number of training runs would
resolve a finer difference.</p>
"""

GROUPS = GROUPS.replace("@ADAPTERCLAUSE@", _adapter_clause())
GROUPS = _fill(GROUPS, _method_tokens())

SUPP_SETUP = SUPP_SETUP.replace("{{MULTIPLICITY}}", _multiplicity_sentence())
SUPP_SETUP = SUPP_SETUP.replace("{{BAND96}}", _band96_clause())
SUPP_SETUP = SUPP_SETUP + _cost_scope()




SUPP_GENERAL = """

<p>Six suites span what an agent recipe could plausibly damage. The available receipts compare
released weights with the earlier LoRA reference checkpoint (<span class="mono">runs/lora/*</span>),
not Table&nbsp;1's current full-parameter stage: commonsense completion
(HellaSwag), coreference (Winogrande), grade-school and elementary science (ARC-e, ARC-c,
OpenBookQA) and broad knowledge (MMLU). Scoring is length-normalised continuation likelihood
over 300 rows per suite, identical rows for the released and the adapted weights, so the
comparison is free of prompt-format and generation effects. The headline is
unchanged: across the eleven baselines the mean change spans &minus;1.8 to +0.7 points, inside
the &plusmn;2.9-point sampling error, and the one family-level regression is Granite-4.0's
5.7&ndash;7.3-point loss on ARC-e.</p>

{{table:pretrain_corpus}}

<p><b>Retraction, and the corrected arm.</b> The five-corpus pretraining comparison was
retracted: shard preparation copied the reference corpus's packed tokens alongside the new
text, so all five arms consumed the original 465M-token corpus &mdash; the corpus-identity
comparison is void, its arms replicates. A correctly packed fresh-token arm at
the same step count reproduces the ToolACE, xLAM and BFCL gains of §4 but gives back 8.1 of
ToolBench's 9.0 points (outside its 6.4 spread), trailing on device control only inside
wide spreads: epochs are optimisation, not memorisation, ToolBench alone is
repetition-keyed, and a one-epoch pair agrees. Two flags: AgentNet collapses for both
initialisations, and Mobile-Actions separates toward the skeleton arm (§4) against its
wide spread.</p>

<p><b>The open-agent pool, and what pretraining cannot import.</b> Three Apache-2.0 corpora
(Toucan-1.5M [42], Hermes [43], Glaive v2 [44]) ask what open agent data buys in
pretraining. Benchmark-embedding sources were rejected; the
one adjacency &mdash; Toucan and MCP-Atlas both draw on real MCP servers &mdash; shares none
of MCP-Atlas's 220 catalog names verbatim. Despite hundreds of millions of genuine MCP
tool-call characters, MCP-Atlas stays at or below its 6.7% chance floor and neither arm beats
the standard mix on any suite. Via SFT,
5,911 Toucan trajectories nudge the <em>novel-tool</em> suites within
spread while device control pays the capacity price (AndroidControl 67.0&rarr;54.1) &mdash;
the clearest claim the 200-row cap manufactured. The xLAM import does clear its spread (+26.8,
99.4% name overlap, nothing elsewhere): imports buy <em>inventory</em> exactly where names
overlap; whether diversity alone buys <em>generalising catalog skill</em> is left open.</p>

<p><b>The staged pipeline: agent data need not enter pretraining at all.</b> Pretrain on
general distillation and dialogue text only (a 743M-token pool: distillation traces [45],
UltraChat [50], hh-rlhf [51]), concentrating the open agentic data (Toucan [42], r0b0t [45]) in
the 700-step midtrain. At 96M, identical two-source SFT gives author/projection scope means
of 77.2/44.7 for curated-web pretraining and 79.0/45.0 for staged pretraining; every per-suite
difference is inside spread. With five-source SFT the corresponding means are 76.8/43.5 and
78.5/46.7. These are descriptive scope means, not a pooled leaderboard rank; no constrained
decode or as-released comparison is used. Doubling SFT steps moves nothing outside spread. At 10.5M and
43M the staged backbone
trails its curated twin (outside spread only on ToolBench at 43M), so Table 1 keeps the
curated ladder. At 96M the pretraining corpus can be whatever general text is cheapest &mdash;
where there is model to absorb it.</p>


<p><b>The vocabulary and context axis.</b> Rebuilt with an LFM-class 65,536-token vocabulary
and 8,192 context (127M parameters, almost entirely embedding table), identical chain and
corpus, the larger vocabulary <em>costs</em> open-catalog accuracy (ToolACE 45.7 against 54.9
at the 16k model, beyond that suite's replicate spread; BFCL down but not beyond it) and 17% of
decode throughput; its one large gain, on AgentNet, sits under that suite's 72.9% majority
floor. Doubling context to 16,384 rescues none of this: ToolACE and BFCL stay down, decode
does not recover, and AgentNet's swing to 33.8 is the floor-dominated instability that keeps
the suite suggestive. At this scale the 16k vocabulary is the better spend on every axis
measured.</p>

""" + _mixer_result()

SUPP_GENERAL = _fill(SUPP_GENERAL, _generic_ladder_numbers())

def _mean_ci_width(stats: dict) -> float:
    """Mean 95% bootstrap width, in points, over the arms that have replicates."""
    widths = []
    for name, entry in stats.get("arms", {}).items():
        if "fcall" in name:
            continue
        low, high = entry["ci95"]
        widths.append((high - low) * 100)
    return sum(widths) / len(widths) if widths else 0.0


def _episode_replicates() -> str:
    """What a training seed is worth on the episode metric, measured rather than imported."""
    path = RESULTS / "analysis/episode_stats.json"
    try:
        stats = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return ""
    replicates = stats.get("replicates") or {}
    usable = {k: v for k, v in replicates.items() if v["runs"] >= 3}
    if len(usable) < 2:
        return ""

    def label(name: str) -> str:
        return name.replace("ep-", "").replace("-fixed", "")

    ordered = sorted(usable.items(), key=lambda kv: kv[1]["spread_points"])
    parts = ", ".join(
        "%s %.3f over %d runs, range %.1f points"
        % (label(name), item["mean"], item["runs"], item["spread_points"])
        for name, item in ordered)
    low = ordered[0][1]["spread_points"]
    high = ordered[-1][1]["spread_points"]
    gap = abs(ordered[-1][1]["mean"] - ordered[0][1]["mean"]) * 100
    verdict = ("inside" if gap <= high else "outside")
    return (
        "<p><b>What a seed is worth.</b> Only two procedures have three seeds: %s. All other "
        "LocalAgent episode cells are single-checkpoint estimates. The replicated procedures' "
        "%.1f&ndash;%.1f-point "
        "ranges are comparable to the roughly %.0f-point task-bootstrap width. The %.1f-point "
        "mean gap is %s the wider seed range and misses the corrected procedure-level test, so "
        "the arms are not ordered. Table&nbsp;1 checkpoints remain zero at every size and seed.</p>"
        % (parts, low, high, _mean_ci_width(stats), gap, verdict))


def _gap_verdict(tight: dict, wide: dict) -> str:
    """Say what the two arms' means do and do not establish, by comparing them."""
    gap = abs(wide["mean"] - tight["mean"]) * 100
    widest = wide["spread_points"]
    if gap <= widest:
        return ("The %.1f-point gap between the two arms' means is inside the wider arm's "
                "own %.1f-point range, so it is not an ordering these runs establish."
                % (gap, widest))
    return ("The %.1f-point gap between their means exceeds that %.1f-point range, but three "
            "seeds per procedure is too few to order the procedures, whatever the fixed-"
            "checkpoint tests say, so the ordering is suggestive rather than established."
            % (gap, widest))


SUPERS = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")


def pfmt(p_value: float) -> str:
    """p-values as m&times;10^k with unicode superscripts, per the editorial convention."""
    if p_value <= 0:
        return "p&nbsp;&lt;&nbsp;10" + "-300".translate(SUPERS)
    import math
    exponent = math.floor(math.log10(p_value))
    mantissa = p_value / (10 ** exponent)
    if exponent <= -10:
        return "p&nbsp;&lt;&nbsp;10" + str(exponent + 1).translate(SUPERS)
    return ("p&nbsp;=&nbsp;%.0f&times;10%s" % (mantissa, str(exponent).translate(SUPERS)))


def _episode_statistics() -> str:
    """Paired tests reported at the two inference units they belong to, never mixed.

    Over tasks with checkpoints fixed, McNemar's exact test compares two specific sets of
    weights; over seeds, three runs per procedure ask whether the training procedures order at
    all. An earlier version of this paragraph mixed the two - asserting that nothing separates
    the arms while its own pair table held Bonferroni-significant checkpoint pairs, and asserting
    interval overlap for a pair whose intervals do not overlap - so every claim here is computed
    from the stats receipt and scoped to its unit.
    """
    path = RESULTS / "analysis/episode_stats.json"
    try:
        stats = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return ""
    pairs = stats.get("pairs", {})
    arms = stats.get("arms", {})
    if not pairs or not arms:
        return ""

    versus_baseline = [v for k, v in pairs.items() if "fcall" in k and "agentic" in k]
    among = {k: v for k, v in pairs.items() if "fcall" not in k}
    sig = {k: v for k, v in among.items() if v.get("significant_bonferroni")}
    sameproc_sig = {k: v for k, v in sig.items() if v.get("same_procedure")}
    if not versus_baseline:
        return ""
    worst_p = max(v["p"] for v in versus_baseline)

    text = (
        "<p><b>Paired tests, at the two units they belong to.</b> Over tasks, checkpoints "
        "fixed, McNemar's exact test under Bonferroni "
        f"across all {len(pairs)} pairs of the LocalAgent grid "
        f"({len(versus_baseline)} arm&ndash;baseline, {len(among)} among arms, "
        f"{len(pairs) - len(versus_baseline) - len(among)} among baselines; the open-family arms "
        "follow a different template and budget and are reported separately) finds all "
        f"arm&ndash;baseline pairs significant (worst {pfmt(worst_p)}) and "
        f"{len(sig)} of the {len(among)} pairs among the arms. ")
    if sameproc_sig:
        example = min(sameproc_sig.values(), key=lambda v: v["p"])
        text += (
            f"{len(sameproc_sig)} of those {len(sig)} are seed replicates of the <em>same</em> "
            f"procedure (smallest {pfmt(example['p'])}, {example['a_only']}/"
            f"{example['b_only']} discordant) &mdash; the decisive caution: task-level "
            "differences between fixed checkpoints are not evidence the training "
            "procedures differ. ")
    reps = stats.get("replicates") or {}
    if len(reps) >= 2:
        parts = ", ".join(
            "%s mean %.3f over %d seeds (range %.1f points)"
            % (name.replace("ep-", "").replace("-fixed", ""), item["mean"], item["runs"],
               item["spread_points"])
            for name, item in sorted(reps.items()))
        text += (
            "Only these two procedures have three seeds; the other LocalAgent arms are "
            "single-checkpoint estimates. Task-level "
            "significance does not order the training procedures once "
            "between-seed variance is considered. Single-checkpoint intervals are "
            "percentile bootstraps over 210 tasks and speak only for that checkpoint.")
    return text + "</p>"


def _episode_cost() -> str:
    """Whether episode supervision cost the Table 1 scores, from the arms' own suite receipts.

    The agentic arms are siblings of the fcall checkpoints - same midtrain parent, same optimiser,
    same budget, same five source corpora, with the episode decisions added - so scoring them on
    the two declared scope groups answers whether the ability was added or traded for. The appendix
    promised this control; this reports it.
    """
    # Every arm with a suite control gets a pair; the fill is dynamic so a control landing later
    # (agenticxl, the 43M rung) joins without an edit here.
    candidates = [("fcall-11m-big", "agentic-11m", "11M base"),
                  ("fcall-40m", "agentic-40m", "43M base"),
                  ("fcall-96m", "agentic-96m", "96M base"),
                  ("fcall-11m-big", "agenticxl-11m", "11M 4&times;"),
                  ("fcall-40m", "agenticxl-40m", "43M 4&times;"),
                  ("fcall-96m", "agenticxl-96m", "96M 4&times;")]
    pairs = [(b, a, l) for b, a, l in candidates
             if report(a, T1_DIR) is not None and report(b, T1_DIR) is not None]
    lines, regressions = [], []
    for base_tag, arm_tag, label in pairs:
        for suite in SUITES:
            a = cell(base_tag, suite, "type_match", T1_DIR)
            b = cell(arm_tag, suite, "type_match", T1_DIR)
            if a is None or b is None:
                continue
            delta = (b - a) * 100
            if abs(delta) > SPREAD.get(suite, 0):
                regressions.append((label, suite, a * 100, b * 100, delta))
        before_author = scope_mean(base_tag, AUTHOR_CALL_SPLITS, directory=T1_DIR)
        after_author = scope_mean(arm_tag, AUTHOR_CALL_SPLITS, directory=T1_DIR)
        before_projected = scope_mean(base_tag, STUDY_PROJECTIONS, directory=T1_DIR)
        after_projected = scope_mean(arm_tag, STUDY_PROJECTIONS, directory=T1_DIR)
        if None not in (before_author, after_author, before_projected, after_projected):
            lines.append("%s author %.1f to %.1f, projection %.1f to %.1f" %
                         (label, before_author, after_author, before_projected, after_projected))
    moved = [r for r in regressions if r[4] < 0]
    gained = [r for r in regressions if r[4] > 0]
    detail = ""
    if moved:
        detail = (" The only losses outside replicate spread: %s."
                  % "; ".join("%s on %s, %.1f to %.1f" % (lab, SUITE_TITLE.get(s, s), a, b)
                              for lab, s, a, b, _ in moved))
    if gained:
        detail += (" Outside spread in the other direction: %s."
                   % "; ".join("%s on %s, %.1f to %.1f" % (lab, SUITE_TITLE.get(s, s), a, b)
                               for lab, s, a, b, _ in gained))
    verdict = ("The 12.5%% arms show no consistent cost across the two groups (the largest "
               "group decline is 0.5 points), while the 38&ndash;40%% arms lower both groups "
               "at the measured sizes: %s." % "; ".join(lines))
    coda = ("The trade therefore follows episode share rather than a single pooled score; the "
            "per-suite exceptions above remain the stronger evidence of interference.")
    return ("<p><b>Sparse shares add; dense shares trade.</b> We compare siblings of the "
            "Table 1 checkpoints: the midtrain parent, optimiser, budget and five source "
            "corpora are fixed, and only the episode decisions are added. "
            "%s%s %s</p>" % (verdict, detail, coda))


def _wilson(payload) -> tuple[float, float]:
    """Wilson 95% interval for one episode receipt, over its 210 tasks."""
    n = len(payload["per_task"])
    rate = payload["episode_success"]
    z = 1.959963985
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5)
    return centre - half, centre + half


def _open_generalisation() -> str:
    """Whether the episode recipe generalises beyond this family, from the open-arm receipts."""
    arms = {}
    for tag in ("openbase-smollm2-135m", "open-smollm2-135m",
                "openbase-qwen3-06b", "open-qwen3-06b"):
        payload = episode_report(f"ep-{tag}-fixed")
        if payload is None:
            return ""
        arms[tag] = payload
    sm_b, sm = arms["openbase-smollm2-135m"], arms["open-smollm2-135m"]
    qw_b, qw = arms["openbase-qwen3-06b"], arms["open-qwen3-06b"]
    sm_h = episode_report("ep-holdout-open-smollm2-135m-fixed")
    qw_h = episode_report("ep-holdout-open-qwen3-06b-fixed")

    def _sc_deltas(model: str) -> tuple[float, float, list[tuple[str, float]]] | None:
        after = report(f"open-ep-{model}", T1_DIR)
        before = report(f"ft-{model}-fullharness", T1_DIR)
        if after is None or before is None:
            return None
        deltas = [(suite, (d["type_match"] - before["suites"][suite]["type_match"]) * 100)
                  for suite, d in sorted(after["suites"].items()) if suite in before["suites"]]
        by_suite = dict(deltas)
        author = sum(by_suite[s] for s in AUTHOR_CALL_SPLITS) / len(AUTHOR_CALL_SPLITS)
        projected = sum(by_suite[s] for s in STUDY_PROJECTIONS) / len(STUDY_PROJECTIONS)
        return author, projected, sorted(deltas, key=lambda x: x[1])
    sc_sm, sc_qw = _sc_deltas("smollm2-135m"), _sc_deltas("qwen3-06b")
    sc_text = ""
    if sc_sm is not None and sc_qw is not None:
        long_name = {"mobileactions": "Mobile-Actions", "toolsandbox": "ToolSandbox",
                     "androidcontrol": "AndroidControl", "toolbench": "ToolBench",
                     "toolace": "ToolACE", "mcpatlas": "MCP-Atlas", "agentnet": "AgentNet",
                     "mind2web": "Mind2Web", "bfcl": "BFCL", "xlam": "xLAM"}
        fmt = lambda v: ("%+.1f" % v).replace("-", "&minus;")
        (w1, d1), (w2, d2) = sc_qw[2][0], sc_qw[2][1]
        rows_note = ", a 17-row suite" if w2 == "toolsandbox" else ""
        sc_text = (" The single-call ledger after episode SFT, on function-name match against "
                   "each model's own Table 1 control: SmolLM2-135M moves %s/%s points on the "
                   "author/projection scope means and Qwen3-0.6B %s/%s, the losses concentrated "
                   "in %s (%s) and %s (%s%s); at 0.6B the 12.5%% share is not free." %
                   (fmt(sc_sm[0]), fmt(sc_sm[1]), fmt(sc_qw[0]), fmt(sc_qw[1]),
                    long_name.get(w1, w1),
                    fmt(d1), long_name.get(w2, w2), fmt(d2), rows_note))
    return (
        "<p><b>The recipe is not ours alone.</b> The same episode decisions at the same 12.5%% "
        "share, added to two open models' own fine-tuning and evaluated through their own chat "
        "templates: SmolLM2-135M moves %.3f&rarr;%.3f and Qwen3-0.6B %.3f&rarr;%.3f. The "
        "zero-without-episodes result generalises &mdash; both controls emit near-perfect calls "
        "(schema %.2f/%.2f, copy fidelity %.2f/%.2f) and complete nothing, choosing the right "
        "tool on only %.0f&ndash;%.0f%% of steps &mdash; and the binding constraint inverts with "
        "scale: the sub-100M arms are limited by verbatim argument fidelity (0.20&ndash;0.34), "
        "the open models by tool choice (%.2f and %.2f agreement after training, at copy "
        "&ge;0.97). Under each family's own post-training configuration (corpus base, "
        "budget and serialisation differ, and the LocalAgent arm consumes roughly ten times "
        "the episode-gradient exposure, 1,500 steps at batch 32 against 600 at batch 8, so "
        "this prices recipes, not architectures) the "
        "10.5M LocalAgent arm attains %.3f against the 135M SmolLM2 arm's %.3f. Each open arm is "
        "one training seed; Qwen3-0.6B's 0.543 carries a task-level Wilson 95%% interval of "
        "[%.3f,&nbsp;%.3f].%s%s</p>"
        % (sm_b["episode_success"], sm["episode_success"],
           qw_b["episode_success"], qw["episode_success"],
           sm_b["step_profile"]["schema_valid"], qw_b["step_profile"]["schema_valid"],
           sm_b["copy_fidelity"]["verbatim"], qw_b["copy_fidelity"]["verbatim"],
           min(sm_b["tool_agreement_vs_gold"], qw_b["tool_agreement_vs_gold"]) * 100,
           max(sm_b["tool_agreement_vs_gold"], qw_b["tool_agreement_vs_gold"]) * 100,
           sm["tool_agreement_vs_gold"], qw["tool_agreement_vs_gold"],
           (episode_report("ep-agentic-11m-fixed") or {}).get("episode_success", 0),
           sm["episode_success"], *_wilson(qw),
           sc_text,
           "" if sm_h is None or qw_h is None else
           " On the fully held-out three-goal composition both trained open arms also complete "
           "nothing (%.3f and %.3f); Qwen3-0.6B reaches %.0f%% of its milestones there without "
           "ever assembling the full sequence, so the compositional ceiling is not an artefact "
           "of our family." % (sm_h["episode_success"], qw_h["episode_success"],
                               qw_h["milestone_rate"] * 100)))


def _oracle_decomposition() -> str:
    """Sequencing against argument fidelity, separated by oracle substitution, from receipts.

    --oracle-mode tools hands the policy the gold plan's tool at every step and keeps its
    arguments: sequencing is free, fidelity is tested. --oracle-mode args does the reverse.
    The paragraph renders only when all six receipts exist, and reports whichever direction
    they actually support.
    """
    arms = (("agenticxl-96m", "best arm"), ("agentic-11m", "11M arm"),
            ("fcall-96m", "Table 1 96M"))
    rows = []
    for tag, label in arms:
        base = episode_report(f"ep-{tag}-fixed") if tag != "fcall-96m" else episode_report("ep-fcall-96m-fixed")
        tools = episode_report(f"ep-oracle-tools-{tag}-fixed")
        args = episode_report(f"ep-oracle-args-{tag}-fixed")
        if base is None or tools is None or args is None:
            return ""
        rows.append((label, base["episode_success"], tools["episode_success"],
                     args["episode_success"]))
    body = "; ".join("%s %.3f free, %.3f sequence given, %.3f arguments given"
                     % row for row in rows)
    best = rows[0]
    seq_gain = best[2] - best[1]
    arg_gain = best[3] - best[1]
    if arg_gain > seq_gain:
        verdict = ("Arguments buy more than sequence: argument fidelity "
                   "binds harder than sequencing on the arm the abstract quotes; planning is "
                   "not exonerated, merely the smaller measured constraint.")
    else:
        verdict = ("Handing over the sequence buys at least as much as the arguments: sequencing "
                   "binds at least as hard as argument fidelity, and the earlier attribution of "
                   "the ceiling to copying alone was wrong.")
    extra = ""
    agree = episode_report("ep-agenticxl-96m-fixed")
    if agree and agree.get("tool_agreement_vs_gold") is not None:
        extra += (" Substitution is post hoc, so the sequence probe could be structurally "
                  "understated (arguments were generated for the model's own tool); two checks "
                  "bound that concern. First, the best arm's freely chosen tool already matches "
                  f"the gold plan's on {agree['tool_agreement_vs_gold'] * 100:.0f}% of steps, so "
                  "the sequence substitution is close to a no-op by construction rather than by "
                  "unfairness.")
    cond = episode_report("ep-oracle-toolscond-agenticxl-96m-fixed")
    if cond:
        extra += (" Second, a conditioned variant states the required tool in the observation and "
                  "lets the policy re-decode the whole call for it: completion is "
                  f"{cond['episode_success']:.3f}, "
                  + ("again indistinguishable from the free run, so the attribution survives "
                     "conditioning." if abs(cond["episode_success"] - rows[0][1]) < 0.05 else
                     "which revises the substitution reading; the receipts carry the split."))
    return ("<p><b>Sequencing against fidelity, separated.</b> <em>Sequence given</em> forces "
            "each step's tool to the gold plan's, keeping the policy's arguments; <em>arguments "
            "given</em> is the reverse. Episode success: %s. %s%s</p>" % (body, verdict, extra))


def _episodes_prose() -> str:
    """Build the episode appendix from the receipts, so its numbers cannot drift from them.

    Every figure quoted here is read out of results/agentic at import time. Clauses whose receipts
    are absent are omitted rather than guessed, which is why the section shortens rather than
    breaks while a sweep is still running.
    """
    bounds = episode_report("free-eval-generated")
    before = episode_report("ep-fcall-11m-big-fixed")
    if not bounds or not before:
        return ""

    oracle = bounds["policies"]["oracle"]
    chance = bounds["policies"]["random"]
    steps = sum(t["steps"] for t in before["per_task"])
    schema_bad = before["errors"].get("schema_invalid", 0)
    # Measured, not inferred. Subtracting the error counts from the step count was wrong: a step
    # whose output does not parse into a call raises no error either, so that difference counted
    # unparseable outputs as successful transitions.
    before_profile = before["step_profile"]
    legal = round(before_profile["transitioned"] * steps)

    text = [
        "<p>An earlier protocol scored these tasks through wrappers that dictate each "
        "step and reported "
        "1.0 &mdash; formatting under dictation. The simulator is sound: "
        "<span class='mono'>_transition</span> enforces preconditions with twelve error codes and "
        "never consults the gold list; <span class='mono'>task_complete</span> is path-"
        "independent containment.</p>",

        "<p><b>The freed harness.</b> The policy sees goal, state and last error; any of the nine device calls from any state; the episode ends when the goal holds or budget is spent. The generated suite: 210 eval tasks, seven families, slot vocabularies partitioned between splits, a quarter starting dirty. Every task replays its gold plan before use; abstention is scored on the action, whose goal holds at step zero.</p>",

        f"<p><b>Bounds.</b> The oracle reaches {oracle['episode_success'] * 100:.0f}% in {oracle['mean_steps']:.1f} steps"
        f", progressing on only {oracle['step_profile']['progressed'] * 100:.0f}% of them; a uniform policy reaches "
        f"{chance['episode_success'] * 100:.1f}%.</p>",

        "{{table:episodes}}",

        f"<p><b>What the Table 1 checkpoint does here.</b> LocalAgent-11M, Table 1's smallest "
        f"model, completes {before['episode_success'] * 100:.1f}% of these episodes and "
        f"spends {before['mean_steps']:.1f} of its 20 steps. The failure is mostly not format "
        f"({before_profile['called_a_tool'] * 100:.0f}% of outputs parse into a call) "
        f"but conditioning on the state. Of {steps:,} steps, "
        f"{schema_bad:,} ({schema_bad / steps * 100:.1f}%) are rejected "
        f"because the arguments belong to a different tool than the one named, and "
        f"only {legal:,} ({before_profile['transitioned'] * 100:.1f}%) change the state at all, "
        f"against the uniform policy's "
        f"{chance['step_profile']['transitioned'] * 100:.0f}%. It selects a plausible tool name and "
        f"argument bag from the goal text without reading the state, reproducing shown text on "
        f"{before['copy_fidelity']['verbatim'] * 100:.0f}% of text-bearing steps: "
        f"the arguments are copied faithfully into the wrong call.</p>",
        _episode_statistics(),
        _episode_replicates(),
        _episode_cost(),
        _oracle_decomposition(),
        _open_generalisation(),
    ]
    after = episode_report("ep-agentic-11m-fixed")
    if after:
        profile = after["step_profile"]
        families = after["per_family"]
        copy = after["copy_fidelity"]
        ranked = ", ".join(
            f"{name.replace('_', '+')} {families[name]['success'] * 100:.0f}%"
            for name in ("recovery", "browser", "notion", "email"))
        text.append(
            f"<p><b>The agentic arm.</b> Adding the episode decisions to the same mixture, "
            f"midtrain parent and 1,500-step budget moves the 11M rung from "
            f"{before['episode_success'] * 100:.1f}% to {after['episode_success'] * 100:.1f}% "
            f"episode success. The state machine is learned: schema-satisfying calls rise from "
            f"{before_profile['schema_valid'] * 100:.1f}% to "
            f"{profile['schema_valid'] * 100:.1f}% of steps, state-moving calls from "
            f"{before_profile['transitioned'] * 100:.1f}% to "
            f"{profile['transitioned'] * 100:.1f}%. Exact reproduction of shown text is not, "
            f"at {copy['verbatim'] * 100:.1f}% over "
            f"{copy['steps_with_text']:,} text-bearing steps; the families rank by "
            f"how much each needs: {ranked}. The dominant error, "
            f"<span class='mono'>send_args_do_not_match_draft</span> "
            f"({after['errors'].get('send_args_do_not_match_draft', 0)}), is text disagreeing "
            f"with the policy's draft. Argument "
            f"fidelity is the largest measured error source; the oracle decomposition "
            f"separates it from sequencing.</p>")

    return "\n\n".join(part for part in text if part)


def table_episodes_main() -> str:
    """Compact episode summary for the main text; the full intervention grid is in App. E."""
    rows_spec = [
        ("ep-fcall-96m-fixed", "Single-call checkpoint, 96M"),
        ("ep-agentic-11m-fixed", "12.5% episodes, 11M"),
        ("ep-agentic-40m-fixed", "12.5% episodes, 43M"),
        ("ep-agentic-96m-fixed", "12.5% episodes, 96M"),
        ("ep-agenticxl-96m-fixed", "38% episodes, 96M"),
    ]
    body = []
    for spec, label in rows_spec:
        if ":" in spec:
            name, pol = spec.split(":")
            payload = episode_report(name)
            payload = payload["policies"][pol] if payload else None
            holdout = None
        else:
            payload = episode_report(spec)
            holdout = episode_report(spec.replace("ep-", "ep-holdout-"))
        if payload is None:
            continue
        profile = payload.get("step_profile", {})
        body.append(row_html([label,
                              f"{payload['episode_success'] * 100:.1f}",
                              f"{profile.get('schema_valid', 0) * 100:.0f}",
                              f"{profile.get('transitioned', 0) * 100:.0f}",
                              "&mdash;" if holdout is None else
                              f"{holdout['episode_success'] * 100:.1f}"]))
    if len(body) < 4:
        return ""
    caption = ("Episode task completion in the stateful simulator (%, 210 tasks). The three "
               "12.5% LocalAgent cells are one checkpoint each, not three-seed means. Schema/Moved: "
               "fraction of steps schema-valid / state-changing. Held-out: a never-trained "
               "three-goal composition. Full profiles, open-model controls, seeds and paired "
               "tests: App. E.")
    header = ("<thead><tr><th>Policy</th><th class='n'>Episode</th><th class='n'>Schema</th>"
              "<th class='n'>Moved</th><th class='n'>Held-out</th></tr></thead>")
    return (f'<table data-caption="{caption}">' + header + "<tbody>"
            + "".join(body) + "</tbody></table>")


SUPP_EPISODES = _episodes_prose()


EPISODES_MAIN = """
<p><b>From a correct call to a completed task.</b> Tables 1 and 2 score one call, but an
agent acts against state produced by its earlier calls. We therefore evaluate policies in a
stateful simulator with real tool preconditions and twelve error codes: the policy sees the
goal, current state and last error, and success is a path-independent check on the final state.
The suite contains 210 held-out tasks over seven families and disjoint slot vocabularies; oracle
replay completes 100.0% and uniform sampling 0.5%. Single-call accuracy does not transfer:
every Table 1 LocalAgent checkpoint completes 0.0%. The 96M checkpoint nevertheless emits a
schema-valid call on 88% of steps and changes state on 50%, showing that legality is not goal
direction. The shared union's 1,606 synthetic multi-turn conversations serialize calls and
replies but omit the simulator's goal/state/last-error decision observation; App. E's episode
rows are state-aligned decisions, which is why the former do not supply this supervision.</p>

{{table:episodes_main}}

<p><b>Episode supervision adds completion, but only inside the trained task family.</b> Re-running
the same fine-tuning with episode decisions at a 12.5% mixture share reaches single-checkpoint
estimates of 20.5%, 34.3% and 17.6% completion at 11M, 43M and 96M. Only the 11M sparse and
96M dense procedures have three-seed repeats; these values establish no size ordering. On the
dataset-author call splits, the descriptive mean
changes +2.3, &minus;0.5 and +1.4 points; on the study projections / filtered views it changes
+1.8, +3.6 and +4.2. Thus there is no consistent single-call cost at the 12.5% share, but one
suite-level regression exceeds its replicate spread (full ledger in App.&nbsp;E), so this is not a
claim of no interference. The same intervention moves SmolLM2-135M from 0.0% to
8.6% and Qwen3-0.6B from 0.0% to 54.3% under their own recipes. A diagnostic on the 38% 96M
arm separates the remaining ceiling: giving the gold tool sequence leaves completion at 25.2%,
whereas giving the gold arguments raises it to 78.6%, making argument fidelity the larger measured
bottleneck in this simulator. Recovery episodes account for 23&ndash;70% of all successful
LocalAgent episodes (42% at 43M), while the explicitly supervised abstention family remains at
0.0% for every reported arm. Every trained checkpoint remains at 0.0% on the never-trained
three-goal composition. These are controlled simulator results, not evidence of real-world or
compositional task completion; protocol, seed variation, paired tests and error profiles are in
App.&nbsp;E.</p>
"""


SUPP_HARNESS = """
<p>Three harness faults changed baseline numbers, each in our favour until fixed.
<b>LFM2-350M</b>'s separate <span class="mono">chat_template.jinja</span> was dropped by a
pattern-limited snapshot, so the harness fell back to plain concatenation and the model scored
0.1% on ToolACE; it also answers in a Pythonic <span class="mono">[name(arg="value")]</span>
form, now parsed. <b>Qwen3-0.6B</b>'s default thinking block consumed the 64-token budget before
any call; the non-thinking path moves ToolACE 0.5%&rarr;88.1%. <b>h2o-danube3-500M</b>'s template
raises on system turns; the harness folds system text into the user turn.</p>

<p><b>Batched decoding.</b> Every model is decoded in batches of 32. For our architecture this is
exactly free: rows are grouped by identical token length (RoPE broadcasts one absolute-position
vector, so left padding would mis-position short rows), and generations are token-identical to
the one-at-a-time path on a 95.3M and a 10.5M checkpoint. For open models it is not free: batch
size changes GEMM reduction order, flipping near-tied early tokens &mdash; over 710 cells, 204
up, 172 down, mean &minus;0.005, mean absolute 0.36 (SmolLM2-135M at most 1.3, Pythia-160M up to
9.9 on Mind2Web). It is a protocol choice, applied to the whole open block; no table mixes the
two (Granite-4.0-H's batch-4 scoring exception: App. B). Re-scoring an unchanged checkpoint reproduces every
suite exactly, so the harness is deterministic under a fixed batch size.</p>
"""

PAPER = {
    "title": "OpenLocalAgent: Transfer and Task Completion in Small Tool-Calling Models",
    "authors": "Sangbum Choi",
    "affiliation": "Toss Bank",
    "abstract": ABSTRACT.strip(),
    "figures": {
        "pipeline": FIGURES / "pf1_pipeline_v2.png",
        "layers": FIGURES / "pf6_layers_main.png",
        "losses": FIGURES / "pf_losses.png",
        "ladder": FIGURES / "pf4_ladder_only.png",
        "headroom": FIGURES / "pf3_headroom.png",
    },
    "tables": {
        "groups": table_groups,
        "ledger": table_ledger,
        "transfer": table_transfer,
        "transfer_full": table_transfer_full,
        "general": table_general,
        "pretrain_corpus": table_pretrain_corpus,
        "pretrain_sources": table_pretrain_sources,
        "train_sources": table_train_sources,
        "answers": table_answers,
        "generic": table_generic,
        "layers": table_layers,
        "causal": table_causal,
        "scale": table_scale,
        "errors": table_errors,
        "general_absolute": table_general_absolute,
        "gaia": table_gaia,
        "clean": table_clean,
        "wide": table_wide,
        "profile": table_profile,
        "donors": table_donors,
        "regions": table_regions,
        "contract": table_contract,
        "corpus": table_corpus,
        "prune": table_prune,
        "suite_sources": table_suite_sources,
        "seeds": table_seeds,
        "constrain": table_constrain,
        "baserecipe": table_baserecipe,
        "rate_asymmetry": table_rate_asymmetry,
        "ftmethod": table_ftmethod,
        "episodes": table_episodes,
        "episodes_main": table_episodes_main,
        "rename": table_rename,
        "noise_sources": table_noise_sources,
        "recipes": table_recipes,
        "mixer": table_mixer,
    },
    "sections": [
        {"title": "Introduction", "body": INTRO},
        {"title": "Related work", "body": RELATED},
        {"title": "One harness, ten public evaluation suites", "body": GROUPS},
        {"title": "Data, weights, or neither: the intervention ledger", "body": LEDGER},
        {"title": "From calls to task completion, and what the recipe costs",
         "body": EPISODES_MAIN},
        {"title": "Stress tests, general cost, and claim boundaries", "body": STRESS_LIMITS},
            ],
    "supplementary": [
        {"title": "Corpus, suites, and further limitations", "body": SUPP_SETUP},
        {"title": "Weight transfer in detail", "body": SUPP_TRANSFER},
        {"title": "General capability, absolute scores", "body": SUPP_GENERAL},
        {"title": "Three harness faults, and what they cost", "body": SUPP_HARNESS},
        *([{"title": "Episodes: acting from the state rather than from a script",
            "body": SUPP_EPISODES}] if SUPP_EPISODES else []),
    ],
    "references": [
        "Liu, W. et al. ToolACE: Winning the Points of LLM Function Calling. arXiv:2409.00920, 2024.",
        "Li, W. et al. On the Effects of Data Scale on UI Control Agents (AndroidControl). "
        "NeurIPS D&B, arXiv:2406.03679, 2024.",
        "Deng, X. et al. Mind2Web: Towards a Generalist Agent for the Web. NeurIPS, 2023.",
        "Zhang, Z. et al. AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement "
        "Fine-Tuning. arXiv:2506.01391, 2025.",
        "Wang, X. et al. OpenCUA: Open Foundations for Computer-Use Agents (AgentNet "
        "dataset). arXiv:2508.09123, 2025.",
        "Liu, Z. et al. APIGen: Automated Pipeline for Generating Verifiable and Diverse "
        "Function-Calling Datasets (xLAM-60k). NeurIPS D&B, arXiv:2406.18518, 2024.",
        "Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.",
        "Allal, L. B. et al. SmolLM2: When Smol Goes Big. arXiv:2502.02737, 2025.",
        "Qwen Team. Qwen2.5 (arXiv:2412.15115, 2024) and Qwen3 (arXiv:2505.09388, 2025) "
        "Technical Reports.",
        "Liquid AI. LFM2 Technical Report. arXiv:2511.23404, 2025.",
        "Pfeiffer, P. et al. H2O-Danube3 Technical Report. arXiv:2407.09276, 2024.",
        "Neyshabur, B., Sedghi, H., Zhang, C. What is Being Transferred in Transfer Learning? "
        "NeurIPS, 2020.",
        "Erdogan, L. E. et al. TinyAgent: Function Calling at the Edge. EMNLP (Demo), "
        "arXiv:2409.00608, 2024.",
        "Belcak, P. et al. Small Language Models are the Future of Agentic AI. "
        "arXiv:2506.02153, 2025.",
        "Shen, W. et al. Small LLMs Are Weak Tool Learners: A Multi-LLM Agent. "
        "arXiv:2401.07324, 2024.",
        "\u017bywot, A. et al. Can Small Agents Collaborate to Beat a Single Large Language "
        "Model? arXiv:2601.11327, 2026.",
        "Qin, Y. et al. ToolLLM: Facilitating Large Language Models to Master 16000+ "
        "Real-world APIs (ToolBench). ICLR, arXiv:2307.16789, 2024.",
        "Chen, T., Goodfellow, I., Shlens, J. Net2Net: Accelerating Learning via Knowledge "
        "Transfer. ICLR, arXiv:1511.05641, 2016.",
        "Chen, C. et al. bert2BERT: Towards Reusable Pretrained Language Models. ACL, "
        "arXiv:2110.07143, 2022.",
        "Wang, P. et al. Learning to Grow Pretrained Models for Efficient Transformer "
        "Training (LiGO). ICLR, arXiv:2303.00980, 2023.",
        "Samragh, M. et al. Scaling Smart: Accelerating LLM Pre-training with Small Model "
        "Initialization (HyperCloning). arXiv:2409.12903, 2024.",
        "Samragh, M. et al. Weight Subcloning: Direct Initialization of Transformers Using "
        "Larger Pretrained Ones. arXiv:2312.09299, 2023.",
        "Kim, Y., Rush, A. M. Sequence-Level Knowledge Distillation. EMNLP, "
        "arXiv:1606.07947, 2016.",
        "Boizard, N. et al. Towards Cross-Tokenizer Distillation: the Universal Logit "
        "Distillation Loss. arXiv:2402.12030, 2024.",
        "Minixhofer, B., Paischer, F., Rekabsaz, N. WECHSEL: Effective Initialization of "
        "Subword Embeddings for Cross-lingual Transfer. NAACL, arXiv:2112.06598, 2022.",
        "Liu, Z. et al. MobileLLM: Optimizing Sub-billion Parameter Language Models for "
        "On-Device Use Cases. ICML, arXiv:2402.14905, 2024.",
        "Chen, W., Li, Z. Octopus v2: On-device Language Model for Super Agent. "
        "arXiv:2404.01744, 2024.",
        "Lin, Q. et al. Hammer: Robust Function-Calling for On-Device Language Models via "
        "Function Masking. ICLR, arXiv:2410.04587, 2025.",
        "Repantis, V. et al. How Many Tools Should an LLM Agent See? A Chance-Corrected "
        "Answer. arXiv:2605.24660, 2026.",
        "Zhang, Q. et al. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient "
        "Fine-Tuning. ICLR, arXiv:2303.10512, 2023.",
        "Ben Zaken, E., Ravfogel, S., Goldberg, Y. BitFit: Simple Parameter-efficient "
        "Fine-tuning for Transformer-based Masked LMs. ACL, arXiv:2106.10199, 2022.",
        "R\u00fcckl\u00e9, A. et al. AdapterDrop: On the Efficiency of Adapters in "
        "Transformers. EMNLP, arXiv:2010.11918, 2021.",
        "Lee, Y. et al. Surgical Fine-Tuning Improves Adaptation to Distribution Shifts. "
        "ICLR, arXiv:2210.11466, 2023.",
        "Panigrahi, A., Saunshi, N., Zhao, H., Arora, S. Task-Specific Skill Localization "
        "in Fine-tuned Language Models. ICML, arXiv:2302.06600, 2023.",
        "Hase, P., Bansal, M., Kim, B., Ghandeharioun, A. Does Localization Inform Editing? "
        "Surprising Differences in Causality-Based Localization vs. Knowledge Editing. "
        "NeurIPS, arXiv:2301.04213, 2023.",
        "IBM Granite Team. Granite 4.0 Language Models. IBM Research, 2025.",
        "Ndubuaku, H. et al. Needle 2: A 45M-Parameter Foundation Tool-Calling Model for "
        "Tiny Devices. Cactus Compute model and software release, 2026.",
        "Patil, S. G., Mao, H., Ji, C. C.-J., Yan, F., Suresh, V., Stoica, I., Gonzalez, J. E. "
        "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation "
        "of Large Language Models. ICML, PMLR 267, 2025.",
        "Lu, J., Holleis, T., Zhang, Y. et al. ToolSandbox: A Stateful, Conversational, "
        "Interactive Evaluation Benchmark for LLM Tool Use Capabilities. arXiv:2408.04682, 2024.",
        "Scale AI. MCP Atlas: Benchmarking Tool Use over Real MCP Servers. Technical report; "
        "dataset ScaleAI/MCP-Atlas (Hugging Face); code scaleapi/mcp-atlas (GitHub), 2025.",
        "Google. Mobile Actions: Android system-tool function-calling traces, the "
        "FunctionGemma-270M training and evaluation set. Dataset google/mobile-actions "
        "(Hugging Face), 2025.",
        "Agent Ark. Toucan-1.5M: 1.5M tool-agent trajectories synthesized from real-world "
        "MCP environments. arXiv:2510.01179; dataset Agent-Ark/Toucan-1.5M (Hugging Face), 2025.",
        "Nous Research and Lambda. Hermes Agent Reasoning Traces: multi-turn tool-calling "
        "trajectories with executed tool results. Dataset lambda/hermes-agent-reasoning-traces "
        "(Hugging Face), 2026.",
        "Glaive AI. Glaive Function Calling v2. Dataset glaiveai/glaive-function-calling-v2 "
        "(Hugging Face), 2023.",
        "r0b0tlab. qwen3.8-max-glm5.2-kimi-k3-distillation: pooled frontier-model distillation traces. Hugging Face dataset, 2026.",
        "Yu, M. et al. The Super Weight in Large Language Models. arXiv:2411.07191, 2024.",
"Biderman, S. et al. Pythia: A Suite for Analyzing Large Language Models Across "
        "Training and Scaling. ICML, 2023.",
        "Gu, A., Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. "
        "COLM, 2024.",
        "Radford, A. et al. Language Models are Unsupervised Multitask Learners (GPT-2). "
        "OpenAI, 2019; Sanh, V. et al. DistilBERT distillation applied as DistilGPT-2, 2019.",
        "Ding, N. et al. Enhancing Chat Language Models by Scaling High-quality Instructional "
        "Conversations (UltraChat). EMNLP, 2023.",
        "Bai, Y. et al. Training a Helpful and Harmless Assistant with RLHF (hh-rlhf). "
        "arXiv:2204.05862, 2022.",
        "Schick, T. et al. Toolformer: Language Models Can Teach Themselves to Use Tools. "
        "NeurIPS, 2023.",
        "Yao, S. et al. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR, 2023.",
        "Patil, S. G. et al. Gorilla: Large Language Model Connected with Massive APIs. "
        "NeurIPS, 2024.",
        "Chen, M. et al. Facilitating Multi-turn Function Calling for LLMs via Compositional "
        "Instruction Tuning (BUTTON). ICLR, arXiv:2410.12952, 2025.",
        "Chai, H. et al. PARL-MT: Learning to Call Functions in Multi-Turn Conversation with "
        "Progress Awareness. arXiv:2509.23206, 2025.",
        "Yao, S., Shinn, N., Razavi, P., Narasimhan, K. tau-bench: A Benchmark for "
        "Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.",
        "Yenugula, R. S. D. P. Wiring Beats Blending: What Transfers Between Transformer "
        "Sizes---and What Doesn't. arXiv:2608.02829, 2026.",
    ],
}
