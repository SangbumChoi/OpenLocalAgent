# Results gallery

Every experiment's plot, in one place. The **why** behind each (motivation, hypothesis, finding) is
in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md). Regenerate any plot with the noted script; new
experiments save here via `from localagent.figs import savefig`.

| # | figure | finding | script |
|---|---|---|---|
| 01 | `01_freegen_vs_grounded.png` | raw byte-generation ~1% vs **grounded constrained decoding 100%** (same 1M model) | `scripts/flywheel.py` |
| 02 | `02_flywheel_accuracy.png` | grounded accuracy per category across flywheel rounds | `scripts/flywheel.py` |
| 03 | `03_pretrain_loss.png` | next-byte pretrain loss | `scripts/flywheel.py` |
| 04 | `04_singleturn_vs_multiturn.png` | single-turn (per category) + multi-turn coding-episode accuracy | `scripts/flywheel.py` |
| 05 | `05_21tools_per_category.png` | 21-tool agent, per-category held-out | `scripts/flywheel.py` |
| 06 | `06_parallel_twocall.png` | learning **parallel two-call** turns (0→38% on 1M) | `scripts/flywheel.py` |
| 07 | `07_failure_driven_flywheel.png` | **failure-driven flywheel**: oversample weak tools → 45→62% | `scripts/analyze_loop.py` |
| 08 | `08_dataset_size_62to71.png` | **more data**: same 1M, 4× dataset → 62→71% | `scripts/analyze_loop.py` |
| 09 | `09_model_size_1M_vs_28M.png` | **model size**: 28M beats 1M overall (+4) and on two-call (+62) | `scripts/analyze_loop.py` |
| 10 | `10_throughput_kvcache.png` | tokens/sec by tier, KV-cache vs recompute | `scripts/benchmark.py` |
| 11 | `11_memory_by_tier.png` | param + KV-cache memory by tier | `scripts/benchmark.py` |
| 12 | `12_tool_retrieval_scale.png` | **retrieval scales to 1,350 tools**; index by example usages | `scripts/tool_scale_analysis.py` |
| 13 | `13_distillation.png` | distillation honest negative on deterministic targets | `scripts/distill_demo.py` |
| 14 | `14_codebench_by_arity.png` | **coding tools**: more args is *easier*; selection is the bottleneck | `scripts/codebench_eval.py` |
| 15 | `15_example_usage_scaling.png` | **how much per-tool data?** 1 example = +16 pts; saturates ~16 | `scripts/example_scaling.py` |
| 16 | `16_scenarios_mcp_rest_cli_sdk.png` | **MCP/REST/CLI/SDK**: it's arg *typing*, not modality (CLI worst at 58%, SDK best) | `scripts/scenarios_eval.py` |
| 17 | `17_logit_entropy_structural_vs_slot.png` | **why grounding wins**: structure bytes 0.18 bits/96.6% top-1, slot bytes 3.28 bits/23.7% — but 75% of slot mass is on prompt bytes (grounding copies them) | `scripts/logit_analysis.py` |
| 18 | `18_multiturn_trajectory_learning.png` | **learning plan→act**: 1M trained on coding+computer-use+planner trajectories, multi-turn step-acc 18%→**89%**, episode-acc→**70%** | `scripts/flywheel.py` |

## The story these tell
- **Grounded/constrained decoding** makes a tiny model reliable (01) — because slot-value bytes are
  high-entropy and its argmax is usually wrong, while the right bytes sit in the prompt (17); the
  **flywheel** improves it (02, 07).
- **Data and model size buy different things**: data fixes data-starved tools (08), size fixes
  confusable ones (09); two-call needed both + a head fix (06).
- **Selection, not grounding, is the bottleneck** at scale (12, 14) — and you need surprisingly
  little per-tool data to retrieve well (15: one example ≫ none).
- It **runs efficiently** on CPU with a KV cache (10, 11).
