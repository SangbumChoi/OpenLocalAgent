"""Dump raw LFM2 generations: a low parse rate is only a finding if the model is not simply
answering in a different call format than the parser accepts."""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from eval_suite import SUITES, HuggingFaceAdapter, _with_suite_catalog, build_tasks, parse_call

adapter = HuggingFaceAdapter("data/baselines/LFM2-350M", "cuda")
for suite in ("toolace", "androidcontrol"):
    tasks = _with_suite_catalog(build_tasks(SUITES[suite], 4))
    print("=" * 24, suite)
    for task in tasks[:4]:
        raw = adapter.predict(task, 64)
        print("GOLD:", task.gold_name, "| PARSED:", parse_call(raw))
        print("RAW:", repr(raw)[:500])
