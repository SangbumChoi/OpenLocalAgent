# Documentation

## Start here

| Document | What it answers |
|---|---|
| [CONVENTIONS.md](CONVENTIONS.md) | Where does anything live, and what do I touch to add a tier / suite / data source / stage? |
| [STAGES.md](STAGES.md) | What are pretrain, midtrain and posttrain each for, and how do I run and resume one? |
| [DATASETS.md](DATASETS.md) | What is every corpus, how big is it, what licence, and how is contamination handled? |
| [BENCHMARKS.md](BENCHMARKS.md) | What are the ten suites, what do the metrics mean, and where does the ladder stand? |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How is the model and pipeline put together? |
| [SPARSE_EXPERTS_REAL_DATA.md](SPARSE_EXPERTS_REAL_DATA.md) | The sparse-expert experiment and the matched-control convention it set. |
| [TOOL_CALLING.md](TOOL_CALLING.md) | The `ToolCaller` API for schema-valid calls on arbitrary tools. |
| [CAMPAIGN.md](CAMPAIGN.md) | What the pre-refactor campaign was, what was removed, and how to recover it. |

`figures/` is the committed plot gallery. Everything written by
`openlocalagent.figs.savefig` lands there, so documents can reference a stable path.

## campaign/

Research notes, debates, runbooks and paper material from the experiment campaign that ran before
the refactor. Kept for provenance — they record why decisions were made and what was measured — but
**they describe the repository as it was, not as it is.** Where one contradicts a document above,
the document above is current.

The one exception worth reading is [SPARSE_EXPERTS_REAL_DATA.md](SPARSE_EXPERTS_REAL_DATA.md),
promoted back to the top level because it governs the sparse arm currently on the ladder: it set
the matched-active convention that `la-150m` exists to satisfy, and it records that the ONNX
exporter cannot lower dynamic expert routing — which bounds what a MoE result can claim.

`campaign/paper/` holds the frozen benchmark plans, acquisition receipts and per-milestone result
JSON that the campaign's numbers rest on. It is large and is not read by the current pipeline.
