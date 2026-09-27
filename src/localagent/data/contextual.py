"""Referent-conditioned dispatch data (same instruction -> different tool by referent type).

Why this module exists: ``paraphrase.py`` teaches MANY phrasings per tool, but each phrasing
still belongs to one tool. Real queries are ambiguous at the verb: "Open X" must route to
``open_url`` when X is a URL, ``open_app`` when X is an app name, and ``read_file`` when X is a
local path. The selector can only learn this if training pairs the SAME instruction skeleton
with different referent types and different gold tools. Each ``GROUPS`` entry is one such
ambiguity class: shared skeletons ("Open {v}.") crossed with per-tool referent branches.

Contracts honored (other code depends on these):
  * Reuses the existing ``Sample`` dataclass and ``_tool_target`` canonical-JSON helper from
    ``agent_synth`` — schema/config untouched. Targets stay compact sorted-key JSON.
  * Every argument value is a LITERAL substring of ``prompt`` (pointer-copy head). URLs and
    paths appear verbatim; quoted-format args are single-quoted inside the referent wording.
  * Train vs eval are DISJOINT in BOTH phrasing skeletons (and referent wraps) AND slot values.
    Train pools here also avoid the eval pools of ``agent_synth``/``paraphrase`` so their
    held-out evals stay honest.

Public API (consumed by retrain code, e.g. ``scripts/train_dispatch_long.py``):
  * ``contextual_samples(n=1, seed=0, split="train") -> list[Sample]`` — ``n`` samples per
    (instruction-group x referent-type) branch, balanced across all branches.
  * ``CONTEXTUAL_EXAMPLES: dict[str, list[str]]`` — per tool, train-pool phrasings drawn FROM
    the ambiguous groups, to enrich example-embeddings with exactly the confusable cases.
"""

from __future__ import annotations

import json
import random

from localagent.data.agent_synth import Sample, _tool_target
from localagent.data.paraphrase import TOOL_SPECS

# ---------------------------------------------------------------------------------------------
# Slot pools — DISJOINT train/eval (module-local). Train values additionally avoid the EVAL
# pools of agent_synth/paraphrase (e.g. no "*.toml" glob, no "npm run dev" command, no
# "Obsidian" app in train) so those held-out evals are not contaminated by this corpus.
# ---------------------------------------------------------------------------------------------

PAGE_URLS = {
    "train": ["https://acme.io/blog", "https://blog.dev/intro-post", "news.site.org/today",
              "https://docs.product.com/guide", "status.host.io/incidents",
              "https://portal.app/changelog"],
    "eval": ["https://forum.example.org/thread/42", "media.host.com/gallery",
             "https://support.vendor.com/faq"],
}
API_URLS = {
    "train": ["https://api.acme.io/v1/users", "api.shop.co/v2/items",
              "https://api.service.dev/orders", "https://data.portal.app/metrics",
              "api.blog.dev/posts"],
    "eval": ["https://api.vendor.net/v1/tickets", "data.metrics.io/summary",
             "https://api.forum.org/threads"],
}
DL_URLS = {
    "train": ["https://cdn.acme.io/release.zip", "files.depot.dev/report.pdf",
              "https://assets.shop.co/logo.png", "mirror.host.dev/tool.deb"],
    "eval": ["https://cdn.vendor.net/installer.dmg", "downloads.forge.io/patch.zip"],
}
LOCAL_FILES = {
    "train": ["src/utils.py", "README.md", "data/notes.txt", "config/settings.yaml",
              "docs/intro.md", "lib/helpers.js"],
    "eval": ["src/router.ts", "docs/guide.md", "data/loader.py"],
}
DIR_PATHS = {
    "train": ["src", "scripts", "assets/icons", "packages", "internal", "notebooks"],
    "eval": ["build", "artifacts", "deploy"],
}
APP_NAMES = {
    "train": ["Spotify", "Terminal", "Slack", "Figma", "Chrome", "Notion"],
    "eval": ["Obsidian", "Postman", "iTerm"],
}
WEB_TOPICS = {
    "train": ["cheap flights to Lisbon", "standing desk reviews", "beginner sourdough recipes",
              "the new metro timetable", "winter running shoes", "local coworking spaces"],
    "eval": ["ergonomic keyboard reviews", "weekend pottery classes",
             "secondhand camera prices"],
}
GREP_TERMS = {
    "train": ["init_db", "AuthMiddleware", "retry_policy", "parse_headers", "MAX_BUFFER",
              "render_sidebar"],
    "eval": ["flush_cache", "RateLimiter", "decode_token"],
}
GLOB_PATTERNS = {
    "train": ["*.css", "test_*.py", "**/*.java", "*.ipynb"],
    "eval": ["*.gradle", "**/*.scss", "conftest_*.py"],
}
# run_python takes ACTUAL inline Python code, never a script FILENAME (running a `.py` file is a
# shell command — see the `python X.py` entries in SHELL_COMMANDS below). Disjoint train/eval.
PY_SNIPPETS = {
    "train": ["print('ok')", "sum(range(8))", "len([1, 2, 3])", "max(4, 9)"],
    "eval": ["min(5, 2)", "abs(-3)"],
}
# Shell commands for run_command. Running a script is `python X.py` (a shell invocation), so those
# live here — NOT in run_python. Disjoint train/eval.
SHELL_COMMANDS = {
    "train": ["make deploy", "git fetch --all", "kubectl get pods", "docker build .",
              "python demo.py", "python scripts/ingest.py"],
    "eval": ["docker ps", "make lint", "git pull", "python migrate.py"],
}
WRITE_DESTS = {
    "train": ["notes/summary.md", "out/results.json", "reports/q1.txt", "logs/run.log"],
    "eval": ["notes/recap.md", "out/metrics.csv"],
}
SLACK_NOTES = {
    "train": ["deploy is done", "meeting moved to 3pm", "release is live", "demo went well"],
    "eval": ["rollback complete", "kickoff starts soon"],
}
NOTION_NOTES = {
    "train": ["ideas for the offsite", "q3 planning draft", "interview feedback",
              "this week's wins"],
    "eval": ["retro action items", "ideas backlog"],
}
CITIES = {
    "train": ["Oslo", "Lisbon", "Vienna", "Stockholm"],
    "eval": ["Riga", "Reykjavik", "Tirana"],
}

# ---------------------------------------------------------------------------------------------
# Instruction groups. Each group: shared "skeletons" (the near-identical instruction wording,
# DISJOINT train/eval) crossed with "branches" (one per referent type / gold tool). A branch's
# "wraps" render the referent around the slot value "{u}" (also DISJOINT train/eval); quoted-
# format args carry their single quotes inside the wrap. Arg-less branches ("pool": None) use
# the wrap itself as the referent. The slot value is ALWAYS a literal substring of the prompt.
# ---------------------------------------------------------------------------------------------

GROUPS: dict[str, dict] = {
    # "Analyze/summarize/read/check/look at X" — URL vs API endpoint vs file vs directory.
    "analyze": {
        "skeletons": {
            "train": ["Analyze {v}.", "Summarize {v}.", "Read {v}.", "Check {v}.",
                      "Look at {v}.", "Take a look at {v}.", "Go through {v}.", "Review {v}."],
            "eval": ["Could you analyze {v}?", "Give me a quick summary of {v}.",
                     "Inspect {v}.", "Have a look at {v}.", "Look over {v}."],
        },
        "branches": [
            {"tool": "open_url", "arg": "url", "pool": PAGE_URLS,
             "wraps": {"train": ["{u}", "the article at {u}"], "eval": ["the post at {u}"]}},
            {"tool": "http_request", "arg": "url", "pool": API_URLS,
             "wraps": {"train": ["the JSON from {u}", "the API response from {u}",
                                 "what a GET to {u} returns"],
                       "eval": ["the JSON payload at {u}", "the response from a GET to {u}"]}},
            {"tool": "read_file", "arg": "path", "pool": LOCAL_FILES,
             "wraps": {"train": ["the file {u}", "{u}"], "eval": ["the local file {u}"]}},
            {"tool": "list_dir", "arg": "path", "pool": DIR_PATHS,
             "wraps": {"train": ["the {u} directory", "what's inside {u}",
                                 "the contents of {u}"],
                       "eval": ["the folder {u}", "what's in the {u} directory"]}},
        ],
    },
    # "Open X" — URL vs desktop app vs local file.
    "open": {
        "skeletons": {
            "train": ["Open {v}.", "Open up {v}.", "Can you open {v}?", "Please open {v}.",
                      "Go ahead and open {v}."],
            "eval": ["Could you open {v}?", "Open {v} for me.", "Pop open {v}."],
        },
        "branches": [
            {"tool": "open_url", "arg": "url", "pool": PAGE_URLS,
             "wraps": {"train": ["{u}", "the page {u}"], "eval": ["the link {u}"]}},
            {"tool": "open_app", "arg": "name", "pool": APP_NAMES,
             "wraps": {"train": ["'{u}'", "the '{u}' app"], "eval": ["the app '{u}'"]}},
            {"tool": "read_file", "arg": "path", "pool": LOCAL_FILES,
             "wraps": {"train": ["the file {u}", "{u}"], "eval": ["the local file {u}"]}},
        ],
    },
    # "Search for X (where)" — web vs codebase grep vs filename glob.
    "search": {
        "skeletons": {
            "train": ["Search for {v}.", "Do a search for {v}.", "Run a search for {v}.",
                      "Can you search for {v}?", "I need you to search for {v}."],
            "eval": ["Could you search for {v}?", "Please search for {v}.",
                     "Go search for {v}."],
        },
        "branches": [
            {"tool": "web_search", "arg": "query", "pool": WEB_TOPICS,
             "wraps": {"train": ["{u} online", "{u} on the web", "{u}"],
                       "eval": ["{u} on the internet"]}},
            {"tool": "grep_search", "arg": "pattern", "pool": GREP_TERMS,
             "wraps": {"train": ["'{u}' in the repo", "'{u}' in the codebase",
                                 "'{u}' in the files"],
                       "eval": ["'{u}' across the codebase", "'{u}' in the source files"]}},
            {"tool": "find_files", "arg": "pattern", "pool": GLOB_PATTERNS,
             "wraps": {"train": ["files named '{u}'", "files matching '{u}'"],
                       "eval": ["any files matching '{u}'"]}},
        ],
    },
    # "Run X" — inline Python CODE (run_python) vs a shell command incl. `python X.py`
    # (run_command) vs the test suite (run_tests). A script FILENAME is a shell command, so it
    # lives in the run_command branch's pool, NOT in run_python.
    "run": {
        "skeletons": {
            "train": ["Run {v}.", "Go run {v}.", "Run {v} now.", "Please run {v}.",
                      "Can you run {v}?"],
            "eval": ["Could you run {v}?", "Run {v} for me.", "Kick off {v}."],
        },
        "branches": [
            {"tool": "run_python", "arg": "code", "pool": PY_SNIPPETS,
             "wraps": {"train": ["the Python snippet '{u}'", "'{u}' in Python"],
                       "eval": ["'{u}' through Python"]}},
            {"tool": "run_command", "arg": "command", "pool": SHELL_COMMANDS,
             "wraps": {"train": ["the command '{u}'", "'{u}'"],
                       "eval": ["the shell command '{u}'"]}},
            {"tool": "run_tests", "arg": None, "pool": None,
             "wraps": {"train": ["the tests", "the test suite", "the unit tests"],
                       "eval": ["all the tests", "the whole test suite"]}},
        ],
    },
    # "Get/fetch X" — API JSON vs download-and-save vs general info lookup.
    "fetch": {
        "skeletons": {
            "train": ["Get {v}.", "Fetch {v}.", "Go get {v}.", "Please fetch {v}.",
                      "Can you get {v}?"],
            "eval": ["Could you fetch {v}?", "Could you get {v}?", "Go fetch {v}."],
        },
        "branches": [
            {"tool": "http_request", "arg": "url", "pool": API_URLS,
             "wraps": {"train": ["the JSON from {u}", "the API response at {u}"],
                       "eval": ["the JSON at {u}"]}},
            {"tool": "download_file", "arg": "url", "pool": DL_URLS,
             "wraps": {"train": ["the file at {u} and save it", "{u} and save it to disk",
                                 "a copy of {u} onto disk"],
                       "eval": ["the file from {u} and save it locally"]}},
            {"tool": "web_search", "arg": "query", "pool": WEB_TOPICS,
             "wraps": {"train": ["me info about {u}", "me some information on {u}"],
                       "eval": ["me the details on {u}"]}},
        ],
    },
    # "Write X to Y" — file path vs Slack channel vs notes/Notion.
    "write": {
        "skeletons": {
            "train": ["Write {v}.", "Can you write {v}?", "Please write {v}.",
                      "Go ahead and write {v}."],
            "eval": ["Could you write {v}?", "Write {v} for me."],
        },
        "branches": [
            {"tool": "write_file", "arg": "path", "pool": WRITE_DESTS,
             "wraps": {"train": ["the summary to {u}", "the results out to {u}",
                                 "this to the file {u}"],
                       "eval": ["the recap to {u}", "the output to the file {u}"]}},
            {"tool": "slack_send", "arg": "message", "pool": SLACK_NOTES,
             "wraps": {"train": ["'{u}' to the #general channel", "'{u}' to Slack",
                                 "'{u}' in the #dev channel"],
                       "eval": ["'{u}' to the #ops channel", "'{u}' over on Slack"]}},
            {"tool": "notion_write", "arg": "content", "pool": NOTION_NOTES,
             "wraps": {"train": ["'{u}' to my notes", "'{u}' to my Notion page",
                                 "'{u}' into Notion"],
                       "eval": ["'{u}' down in my notes", "'{u}' onto my Notion page"]}},
        ],
    },
    # "Check X" — weather vs repo status vs pending diff vs a URL.
    "check": {
        "skeletons": {
            "train": ["Check {v}.", "Check on {v}.", "Can you check {v}?",
                      "Check {v} real quick.", "Go check {v}."],
            "eval": ["Could you check {v}?", "Check {v} for me.", "Please check {v}."],
        },
        "branches": [
            {"tool": "get_weather", "arg": "city", "pool": CITIES,
             "wraps": {"train": ["the weather in {u}", "the forecast for {u}"],
                       "eval": ["today's weather in {u}", "the weather over in {u}"]}},
            {"tool": "git_status", "arg": None, "pool": None,
             "wraps": {"train": ["the repo", "git status", "the state of the repo"],
                       "eval": ["the repo status", "the working tree status"]}},
            {"tool": "git_diff", "arg": None, "pool": None,
             "wraps": {"train": ["what changed", "the diff", "what's changed in the code"],
                       "eval": ["the latest diff", "what changed in the working tree"]}},
            {"tool": "open_url", "arg": "url", "pool": PAGE_URLS,
             "wraps": {"train": ["{u}", "the page {u}"], "eval": ["the page at {u}"]}},
        ],
    },
}


def _render(skeleton: str, branch: dict, rng: random.Random, split: str, group: str) -> Sample:
    """One Sample: shared instruction skeleton x this branch's referent type."""
    wrap = rng.choice(branch["wraps"][split])
    if branch["pool"] is None:                         # arg-less referent (e.g. "the tests")
        referent, args = wrap, {}
    else:
        value = rng.choice(branch["pool"][split])
        referent = wrap.replace("{u}", value)          # value stays a literal substring
        args = {branch["arg"]: value}
    name = branch["tool"]
    prompt = skeleton.replace("{v}", referent)
    return Sample(f"ctx_{group}", TOOL_SPECS[name]["group"], prompt, "tool",
                  _tool_target(name, args), name,
                  json.dumps(args, separators=(",", ":"), sort_keys=True))


def contextual_samples(n: int = 1, seed: int = 0, split: str = "train") -> list[Sample]:
    """Referent-conditioned dispatch Samples, ``n`` per (instruction-group x referent) branch.

    Balanced: every branch of every group contributes exactly ``n`` samples, so
    ``len == n * sum(len(g["branches"]) for g in GROUPS.values())``. ``split="eval"`` draws
    DISJOINT skeletons, referent wraps, and slot values — it measures whether dispatch is truly
    conditioned on the referent type rather than memorized surface strings.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = random.Random(seed)
    out: list[Sample] = []
    for gname, gspec in GROUPS.items():
        skeletons = gspec["skeletons"][split]
        for branch in gspec["branches"]:
            for _ in range(n):
                out.append(_render(rng.choice(skeletons), branch, rng, split, gname))
    rng.shuffle(out)
    return out


# Train-pool example phrasings per tool, FROM the ambiguous groups — seeds example-embeddings
# with exactly the confusable cases (e.g. open_url and read_file both get "Open ..." examples).
def _build_examples() -> dict[str, list[str]]:
    ex: dict[str, list[str]] = {}
    for gspec in GROUPS.values():
        skeletons = gspec["skeletons"]["train"][:3]
        for branch in gspec["branches"]:
            wraps = branch["wraps"]["train"]
            pool = None if branch["pool"] is None else branch["pool"]["train"]
            for i, skeleton in enumerate(skeletons):
                wrap = wraps[i % len(wraps)]
                referent = wrap if pool is None else wrap.replace("{u}", str(pool[i % len(pool)]))
                phrase = skeleton.replace("{v}", referent)
                bucket = ex.setdefault(branch["tool"], [])
                if phrase not in bucket:
                    bucket.append(phrase)
    return ex


CONTEXTUAL_EXAMPLES: dict[str, list[str]] = _build_examples()
