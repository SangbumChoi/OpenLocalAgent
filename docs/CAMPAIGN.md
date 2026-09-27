# The campaign, and what happened to it

Before the refactor this repository held about 250,000 lines: a long experiment campaign had
accreted around a much smaller working core. The counts, for scale:

| | Before | After |
|---|---|---|
| `src/openlocalagent` modules | 143 (109k LOC) | 91 (62k LOC) |
| `scripts/` | 396 (121 of them `assemble_m*.py`) | 51 |
| `launchers/` | 197 | 0 |
| `tests/` | 582 (358 of them `test_m*.py`) | 92 |
| `configs/` | 622 | 71 |

The documentation followed the same shape. `docs/campaign/` now holds the research notes, debates,
runbooks and paper material — 1,451 files including every per-milestone result JSON. They record
why decisions were made and what was measured, and they describe the repository as it was. Where
one contradicts a document at `docs/` top level, the top-level document is current.

Nothing removed was reachable from an entry point any more. What it *was* doing was hiding the
parts that are: the three training stages and the ten-benchmark eval suite.

## Recovering any of it

Every file is in git under the `pre-refactor` tag.

```bash
git show pre-refactor:scripts/assemble_m650_appworld_api_head.py
git show pre-refactor --stat | head -50
git checkout pre-refactor -- launchers/          # bring a whole directory back
```

`git log --diff-filter=D --name-only pre-refactor..HEAD` lists everything deleted and in which
commit, with the reasoning in that commit's message.

## What was deliberately kept

- **The ten-benchmark eval suite**, moved from `scripts/eval_suite.py` into
  `src/openlocalagent/eval/suite.py`. It produced every number in `results/evalsuite-full/`.
- **`results/`** — the receipts the published comparisons rest on.
- **`docs/figures/`** — the plots those receipts produced.
- **The data ingestion and normalisation scripts** for all ten suites, since a suite you cannot
  rebuild is a suite you cannot check.

## What the campaign is still good for

`results/evalsuite-full/` is the baseline table the current ladder is measured against — see
[BENCHMARKS.md](BENCHMARKS.md). The open-model rows there all received the same fine-tuning recipe,
which is what makes "does our from-scratch model beat a fine-tuned open one at this size" a fair
question rather than a rhetorical one.

## The lesson the campaign paid for

From the old `launchers/box/README.md`, kept here because it is the reason the eval suite now lives
in the library:

> the `xlam_opaque` and `xlam_shuffled` evaluation suites were defined only in the box's copy of
> `scripts/eval_suite.py`, and syncing this tree's older copy over it deleted them. The loss was
> silent because the suite loop skipped unrecognised `--suites` names without complaining, so the
> run that followed exited `rc=0` and wrote a receipt missing the suite it was asked for. Two
> receipts had to be discarded as unreproducible.

Anything that generates a number this project reports belongs in version control, and a tool asked
for something it does not recognise should fail rather than continue quietly.
