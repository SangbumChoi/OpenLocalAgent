# Ladder demo page

The status page for the training ladder: three stage blocks (input → process → output) with the
per-step loss curve inside each process column, then the ten-benchmark results.

`page.template.html` is the page with a `__DATA__` placeholder. `build.py` fills it from live run
data and writes a publishable file.

```bash
python demos/ladder/build.py --runs runs --results results/evalsuite-full --out /tmp/ladder.html
```

It reads `curve.jsonl` from every `runs/<stage>-la-<tier>/` and every
`results/evalsuite-full/*.json`, downsamples the curves to ~200 points each, and inlines the
result as JSON. Nothing is fetched at view time, so a published page is a snapshot — republish to
refresh it.

Kept in the repository rather than in a scratch directory for the reason `docs/CAMPAIGN.md`
records: the code that produces a number this project shows belongs in version control.
