#!/usr/bin/env python3
"""Submission preflight: everything that must hold before the PDF leaves the building.

Run from docs/paper/slmw2026. Exits non-zero on any failure, so it can gate the final build.
Each check exists because its failure mode actually happened this campaign: numbers outliving
receipts, entities and tags reaching the PDF, a section silently emptying, the page budget
breaking, and the checklist contradicting the appendix.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    import content

    print("== content generation ==")
    check("ft_method resolves", content.ft_method() in ("lora", "full"),
          repr(content.ft_method()))
    check("no unfilled @TOKEN@ in prose",
          not any(re.findall(r"@[A-Z]+@", v) for k, v in vars(content).items()
                  if isinstance(v, str) and k.isupper()))
    for name in ("table_groups", "table_ledger", "table_episodes", "table_rename",
                 "table_mixer", "table_noise_sources", "table_regions", "table_ftmethod",
                 "table_rate_asymmetry", "table_baserecipe"):
        html = getattr(content, name)()
        check(f"{name} renders", bool(html) and html.count("<tr>") >= 2,
              f"{html.count('<tr>')} rows")
    check("evaluation scope language is explicit",
          bool(re.search(r"ten public\s+evaluation suites", content.ABSTRACT))
          and "ten public benchmarks" not in content.ABSTRACT
          and "study projections / filtered views" in content.table_groups())
    groups = content.table_groups()
    check("Table 1 separates controlled and unmatched panels",
          "Panel A" in groups and "Panel B" in groups
          and "excluded from maxima" in groups
          and all(f"{name} (matched)" in groups for _tag, name, _pre, _matched in content.OURS)
          and all(f"{name} (shipped)" in groups for _tag, name, _pre, _matched in content.OURS))
    check("App. B places every promised fairness table",
          all(f"{{{{table:{name}}}}}" in content.SUPP_TRANSFER
              for name in ("ftmethod", "rate_asymmetry", "baserecipe")))
    rename = content.table_rename()
    check("rename stress test does not cross LoRA/full lineages",
          "Qwen3" not in rename and "SmolLM2" not in rename)
    general = content.table_general()
    general_receipts = [content.report(tag, "general") for tag, _ in content.PUBLIC]
    check("general-cost table labels its LoRA receipt lineage",
          "earlier LoRA reference" in general
          and all(p and str(p.get("adapter", "")).startswith("runs/lora/")
                  for p in general_receipts))
    ledger = content.table_ledger()
    check("Table 2 emphasises large losses as well as gains",
          "<b>-32.4</b>" in ledger and "<b>-9.3</b>" in ledger)
    check("episode seed scope is not overstated",
          "three seeds per arm" not in content.SUPP_EPISODES.lower()
          and re.search(r"single-checkpoint\s+estimates", content.EPISODES_MAIN))
    check("chance claim is scoped to the robust suite",
          "past chance" not in content.ABSTRACT
          and "past chance" not in content.INTRO)
    check("asset licenses stated in paper",
          "Apple Sample Code License" in content.SUPP_SETUP
          and "CC-BY-4.0" in content.SUPP_SETUP)
    audit = content._multiplicity_audit()
    check("multiplicity audit runs", audit["comparisons"] > 0,
          str(audit["comparisons"]))
    check("episode appendix non-empty", len(content.SUPP_EPISODES) > 2000,
          str(len(content.SUPP_EPISODES)))
    import json as _json
    stats = _json.loads((HERE / "results/analysis/episode_stats.json").read_text())
    primary = {k: v["episode_success"] for k, v in stats["arms"].items()
               if "fcall" not in k and "-s1" not in k and "-s2" not in k}
    lo, hi = min(primary.values()) * 100, max(primary.values()) * 100
    check("abstract's 'zero to double-digit' matches primary arms",
          10 <= lo <= hi < 100, f"{lo:.1f}-{hi:.1f}")
    oracle = content.episode_report("ep-oracle-args-agenticxl-96m-fixed")
    check("oracle-args receipt behind the fidelity claim present",
          oracle is not None and abs(oracle["episode_success"] - 0.79) < 0.005,
          "receipt absent" if oracle is None else f"{oracle['episode_success']:.3f}")

    print("== render and build ==")
    r = subprocess.run([sys.executable, "render_latex.py"], cwd=HERE,
                       capture_output=True, text=True)
    check("render_latex", r.returncode == 0, r.stderr[-200:])
    b = subprocess.run(["tectonic", "main.tex"], cwd=HERE / "neurips",
                       capture_output=True, text=True)
    check("tectonic", b.returncode == 0, b.stderr[-200:])

    print("== PDF invariants ==")
    text = subprocess.run(["pdftotext", "-layout", str(HERE / "neurips/main.pdf"), "-"],
                          capture_output=True, text=True).stdout
    pages = text.split("\f")
    ref_page = next((i for i, p in enumerate(pages, 1) if "References" in p), None)
    check("main text fits six pages", ref_page is not None and ref_page <= 7,
          f"References opens page {ref_page}")
    import re as _re
    def page_of(probe):
        return next((i for i, pg in enumerate(pages, 1)
                     if probe in _re.sub(r"\s+", " ", pg)), None)
    app_a = page_of("Corpus, suites")
    checklist = page_of("NeurIPS Paper Checklist")
    if app_a and checklist:
        span = checklist - app_a
        check("appendices within ten pages", span <= 10, f"{span} pages ({app_a}-{checklist - 1})")
    abstract = _re.sub(r"\s+", " ", pages[0])
    a0 = abstract.find("Tool-calling research")
    a1 = abstract.find("1 Introduction")
    if a0 >= 0 and a1 > a0:
        words = len(abstract[a0:a1].split())
        check("abstract under 200 words", words <= 200, f"{words} words")
    leaks = sorted(set(re.findall(r"</?[a-z]+>|&[a-z]+;|@[A-Z]+@|\{\{table", text)))
    check("no entities/tags/placeholders in PDF", not leaks, str(leaks))

    print("== checklist consistency ==")
    checklist = (HERE / "neurips/checklist_filled.tex").read_text()
    check("checklist says code ships as supplementary material",
          "attached as supplementary material" in checklist
          and "not released with this submission" not in checklist)
    check("checklist names Section 6, not Section 8",
          "Section 8" not in checklist and "Section 6" in checklist)
    check("appendix omits release-policy paragraph",
          "Availability, stated exactly" not in content.SUPP_SETUP
          and "released at camera-ready" not in content.SUPP_SETUP)

    print()
    if failures:
        print(f"PREFLIGHT FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("PREFLIGHT CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
