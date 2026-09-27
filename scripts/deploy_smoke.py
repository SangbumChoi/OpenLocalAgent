"""End-to-end local deployment smoke with a machine-readable, honest receipt.

This exercises checkpoint loading, route selection, argument grounding, and ``ToolRegistry``
dispatch.  The registry is an echo stub: no browser, OS, email account, or external tool server is
invoked.  The receipt therefore measures local dispatch behavior only and cannot satisfy a native
benchmark or workshop publication gate.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/tiny-30m-scenarios-best.pt")
ap.add_argument("--out", help="optional no-clobber JSON receipt path")
args = ap.parse_args()

reg = ToolRegistry()
for t in TOOLS:
    reg.register(t, (lambda _n: (lambda **kw: {"ok": _n, "args": kw}))(t.name))

agent = Agent.from_checkpoint(args.ckpt, reg)
print(f"loaded {args.ckpt}: selector={agent.selector is not None} "
      f"route_head={agent.route_head is not None} ptr={agent.ptr_head is not None}\n", flush=True)

CASES = [
    {"id": "text_answer", "category": "text", "prompt": "What is the color of a monkey?", "expected_tool": None},
    {"id": "browser_search", "category": "browser", "prompt": "Look up who invented the telephone.", "expected_tool": "web_search"},
    {"id": "browser_open", "category": "browser", "prompt": "Open https://github.com/pytorch/pytorch in the browser", "expected_tool": "open_url"},
    {"id": "filesystem_list", "category": "computer", "prompt": "List the files in the src directory", "expected_tool": "list_dir"},
    {"id": "code_search", "category": "computer", "prompt": "Search the codebase for the function train_step", "expected_tool": "grep_search"},
    {"id": "filesystem_make_dir", "category": "computer", "prompt": "Make a directory called build", "expected_tool": "make_dir"},
    {"id": "run_tests", "category": "computer", "prompt": "Run the test suite", "expected_tool": "run_tests"},
    {"id": "calculator", "category": "tool_api", "prompt": "What is 18 * 24?", "expected_tool": "calculator"},
    {"id": "email_send", "category": "productivity", "prompt": "Email Dana the quarterly report", "expected_tool": "send_email"},
    {"id": "download", "category": "browser", "prompt": "Download the dataset from https://example.com/data.zip", "expected_tool": "download_file"},
]
TOOL_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\(")
records = []
for case in CASES:
    output = agent.chat(case["prompt"])
    match = TOOL_RE.search(output)
    predicted_tool = match.group(1) if match else None
    exact_tool = predicted_tool == case["expected_tool"]
    records.append(
        {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "expected_tool": case["expected_tool"],
            "predicted_tool": predicted_tool,
            "exact_tool": exact_tool,
            "raw_output": output,
        }
    )
    print(f">> {case['prompt']}\n   {output}", flush=True)
summary = {
    "cases": len(records),
    "exact_tool": sum(record["exact_tool"] for record in records),
    "tool_accuracy": sum(record["exact_tool"] for record in records) / len(records),
    "by_category": {
        category: {
            "cases": sum(record["category"] == category for record in records),
            "exact_tool": sum(record["category"] == category and record["exact_tool"] for record in records),
        }
        for category in sorted({record["category"] for record in records})
    },
}
for category, result in summary["by_category"].items():
    result["tool_accuracy"] = result["exact_tool"] / result["cases"]
receipt = {
    "kind": "localagent_local_deployment_smoke",
    "schema_version": 1,
    "checkpoint": {
        "path": str(Path(args.ckpt)),
        "sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
    },
    "environment": {
        "name": "local_tool_registry_echo_stub",
        "environment_executed": False,
        "external_accounts": False,
        "screenshots": False,
    },
    "summary": summary,
    "cases": records,
    "claim_boundary": (
        "Local checkpoint dispatch smoke only. ToolRegistry handlers echo calls; this is not a "
        "browser, Android, desktop VM, email, Notion, MCP, or official benchmark result."
    ),
}
if args.out:
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"receipt: {output}", flush=True)
print(json.dumps({"kind": receipt["kind"], "summary": summary}, sort_keys=True), flush=True)
print("\nDEPLOY_DONE", flush=True)
