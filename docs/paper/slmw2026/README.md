# NeurIPS 2026 SLM-Agents submission source

This directory contains the double-blind four-page extended-abstract source. It is intentionally
separate from `../SLMW2026_DRAFT.md`, which is the long-form evidence ledger.

The submission thesis and claim boundaries are summarized in
[`../WORKSHOP_INSIGHT.md`](../WORKSHOP_INSIGHT.md). It treats feature-materialization parity as
the central negative deployment finding; the evidence ledger remains authoritative for every
pending result.

## Template provenance

- Official source:
  `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`
- Archive SHA-256:
  `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`
- `neurips_2026.sty` SHA-256:
  `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`
- Downloaded and verified: 2026-07-28

The style file is vendored without modification. The workshop CFP requires the NeurIPS workshop
template and double-blind review. The live CFP permits either a four-page extended abstract plus
references or an optional eight-page full paper; this source targets four content pages.

## Build

From this directory:

```bash
tectonic main.tex
```

The verified working render is copied to
`../../../output/pdf/slmw2026-compact-webgpu-agents-wip.pdf` with SHA-256
`305a6f49ae4e46748964d69acc5f810a11fd80a25536d51d944304df42df5eb9`. The
body and references fit within page 4; the checklist begins on page 5 and continues through
page 11 (11 pages total). Generated PDFs are build artifacts and should not be committed.

The submission is not ready merely because it compiles. Before submission:

1. keep the main content at four pages or fewer without changing the style;
2. turn each conservative checklist `No`/`N/A` into `Yes` only after the stated prerequisite is
   actually satisfied;
3. run the externally timestamped fresh-capability protocol and update only from preserved
   artifacts;
4. package an anonymous code/data supplement and scan it for repository remotes, usernames,
   filesystem paths, PDF metadata, and checkpoint metadata;
5. verify every citation against its primary source and every number against the bound JSON
   artifact.

The Markdown evidence ledger remains authoritative when a result is pending; the LaTeX paper must
not turn a planned experiment into a completed claim.
