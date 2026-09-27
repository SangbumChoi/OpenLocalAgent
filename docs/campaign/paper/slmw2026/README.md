# docs/paper/slmw2026 — the NeurIPS 2026 submission, receipt-driven

`content.py` is the single source: prose, tables and captions are generated from the eval
receipts in `results/`, so a rebuild after new receipts land cannot leave a stale number.

Build (self-contained: run from this directory, no paths outside the repo):
```
python3 render_latex.py && (cd neurips && tectonic main.tex)
```
`render_latex.py` resolves figures and receipts relative to this directory, so the emitted
`neurips/main.tex` embeds repo paths only. The main text must stay at 6 pages: check with
`pdftotext -layout neurips/main.pdf - | grep -n References` — References must open page 7.

- `results/evalsuite-full/` — the full-pool protocol receipts every Table 1/2 cell reads
  (per-suite whole evaluation pools; `-constrained`/`-fallback` suffixes are decode-mode
  pairs of the same checkpoint)
- `results/{throughput,analysis,evalsuite-clean,gaia,general,...}` — the appendix receipt sets
- `figures/` — the three figures the build embeds
- `fig_losscurves.py` — regenerates `figures/pf_losses.png` from
  `results/analysis/losscurves_perstep.json` (per-step training loss + validation markers)
- `neurips/main.pdf` — current build (main text 6pp)
- `korean_translation.md` — 국문 번역본 (tables regenerated from the same receipts)

The earlier SLM-Agents workshop draft this directory previously held is in git history
(removed when this became the main paper tree).
