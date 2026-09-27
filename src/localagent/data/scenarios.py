"""SOTA-agent SCENARIO corpus (Codex / Claude Code behaviour families).

Why this module exists: ``paraphrase.py`` and ``contextual.py`` teach *which tool* a single
request maps to. But modern coding/research agents also have to decide *when NOT to act yet*
(ask a clarifying question), *when not to act at all* (over-trigger negatives), *when to fan out*
(parallel calls), and *how to carry data across turns* (chained / error-recovery trajectories).
This source adds exactly those four single-turn behaviours and three multi-turn families on top
of the existing generators, reusing their primitives verbatim.

Contracts honored (other code depends on these):
  * Reuses the existing ``Sample`` dataclass and ``_tool_target`` canonical-JSON helper, and the
    schema ``Conversation``/``Message``/``Role``/``ToolCall`` — schema/configs/render untouched.
  * Parallel samples carry the SAME shape ``agent_synth.parallel`` produces: ``calls`` is a list
    of ``{"name","arguments"}`` dicts (len >= 2), ``target`` is the space-joined ``_tool_target``
    of each call, and ``ref_name``/``ref_args`` are the FIRST call.
  * Pointer-copy grounding: every tool-call arg value that is COPIED appears as a literal
    substring of the relevant prior turn — the USER prompt for a first call, and the prior
    ``tool_response`` for chained / error-recovery follow-ups.
  * All tools referenced are in ``localagent.agent.toolset.STANDARD_TOOLS``.
  * Train vs eval are DISJOINT in BOTH phrasing skeletons AND slot values.

Public API:
  * ``scenario_samples(n=1, seed=0, split="train") -> list[Sample]`` — single-turn additions,
    balanced across {clarify, abstain, parallel}; ``n`` is the count PER sub-type.
  * ``scenario_episodes(n=1, seed=0, split="train") -> list[Conversation]`` — multi-turn
    episodes, balanced across {workflow, chained, error_recovery}; ``n`` is the count PER family.
"""

from __future__ import annotations

import json
import random

from localagent.data.agent_synth import Sample, _tool_target
from localagent.data.schema import Conversation, Message, Role, ToolCall

# ---------------------------------------------------------------------------------------------
# Slot pools — DISJOINT train/eval (module-local, so the eval pool here never overlaps train).
# Values ground as literal substrings under any skeleton; quoted-format args carry their quotes
# inside the phrasing, paths/urls appear verbatim.
# ---------------------------------------------------------------------------------------------

# code paths (read_file / edit_file / grep targets). Disjoint train/eval.
S_PATHS = {
    "train": ["app/core.py", "lib/parse.py", "src/server.ts", "pkg/store.go", "web/view.jsx",
              "utils/net.py", "api/users.py", "services/cart.py", "db/migrate.sql", "cli/main.rs",
              "core/queue.py", "lib/cache.ts", "src/router.go", "web/form.jsx", "api/orders.py",
              "services/mailer.py", "pkg/codec.go", "cli/args.rs"],
    "eval": ["mod/auth.py", "ui/panel.tsx", "svc/billing.py", "net/socket.go", "data/etl.py",
             "mod/session.py", "ui/modal.tsx", "svc/ledger.py", "net/dialer.go"],
}
# grep patterns / symbols. Disjoint train/eval.
S_PATTERNS = {
    "train": ["compute_total", "RetryQueue", "parse_token", "MAX_RETRIES", "load_config",
              "render_row", "validate_input", "build_index", "flush_buffer", "open_session",
              "encode_frame", "WorkerPool", "split_chunk", "DEFAULT_TTL", "resolve_path"],
    "eval": ["decode_frame", "RateGuard", "merge_shards", "MIN_BATCH", "warm_cache",
             "drain_queue", "ShardMap", "trim_prefix"],
}
# shell commands (run_command). Disjoint train/eval.
S_COMMANDS = {
    "train": ["make build", "npm run compile", "cargo check", "go vet ./...", "make migrate",
              "yarn lint", "tox -e py311", "pnpm build", "make docs", "go build ./..."],
    "eval": ["make package", "npm run typecheck", "cargo clippy", "yarn test", "make release"],
}
# commit messages (git_commit). Disjoint train/eval.
S_COMMITS = {
    "train": ["fix null deref", "tighten validation", "cache the lookup", "guard empty input",
              "split the helper", "log the retry", "drop dead branch", "rename the field"],
    "eval": ["debounce the click", "trim trailing space", "pin the toolchain",
             "inline the constant", "wrap the error"],
}
# research queries (web_search). Disjoint train/eval.
S_QUERIES = {
    "train": ["rust async runtimes", "postgres index types", "kafka consumer groups",
              "wasm component model", "grpc streaming patterns", "sqlite wal mode",
              "nginx reverse proxy", "elixir supervision trees", "terraform state locking"],
    "eval": ["zig comptime basics", "duckdb vs sqlite", "redis stream consumers",
             "envoy circuit breaking", "clickhouse merge trees"],
}
# result URLs returned by a search (open_url follow-up). Disjoint train/eval.
S_URLS = {
    "train": ["docs.rs/tokio", "wiki.postgresql.org/Indexes", "kafka.apache.org/intro",
              "webassembly.org/specs", "grpc.io/docs/guides", "nginx.org/en/docs",
              "hexdocs.pm/elixir", "terraform.io/docs/state"],
    "eval": ["ziglang.org/learn", "duckdb.org/docs", "redis.io/docs/streams",
             "envoyproxy.io/docs", "clickhouse.com/docs"],
}
# cities for parallel weather batches (need >=3 per split for weather3). Disjoint train/eval.
S_CITIES = {
    "train": ["Turin", "Bergen", "Cork", "Aarhus", "Ghent", "Bilbao", "Graz", "Nantes",
              "Porto", "Tampere", "Utrecht", "Malmo"],
    "eval": ["Bratislava", "Cluj", "Leuven", "Aalborg", "Gdansk", "Brno", "Trento"],
}
# directory names for parallel make_dir batches. Disjoint train/eval.
S_DIRS = {
    "train": ["build/cache", "tmp/work", "out/dist", "logs/jobs", "data/raw", "var/state",
              "out/bundle", "tmp/scratch", "data/staging"],
    "eval": ["dist/web", "scratch/run", "cache/blobs", "var/spool", "out/reports"],
}
# package names for parallel install batches. Disjoint train/eval.
S_PKGS = {
    "train": ["lodash", "axios", "vitest", "zod", "esbuild", "pino", "fastify"],
    "eval": ["undici", "valibot", "tsx", "hono"],
}

# Clarify: underspecified request -> short follow-up QUESTION (no tool, kind="text").
# Skeletons disjoint train/eval; the target question is the canonical clarification.
CLARIFY = {
    "train": [
        ("Email the team.", "Who should I email, and what should the message say?"),
        ("Book a meeting.", "What time, and with whom?"),
        ("Schedule a call.", "When should the call be, and who is on it?"),
        ("Create a reminder.", "What should I remind you about, and when?"),
        ("Open the file.", "Which file should I open?"),
        ("Run the script.", "Which script should I run?"),
        ("Make a reservation.", "Where, for how many people, and at what time?"),
        ("Add it to my calendar.", "What is the event, and when is it?"),
        ("Set a timer.", "How long should the timer run for?"),
        ("Install the package.", "Which package would you like me to install?"),
        ("Commit the changes.", "What commit message should I use?"),
        ("Search the codebase.", "What symbol or string should I search for?"),
        ("Download it.", "What is the URL of the file to download?"),
        ("Play something.", "What song or artist would you like to hear?"),
        ("Create a ticket.", "What should the ticket title and description be?"),
        ("Move the file.", "Which file, and where should it go?"),
        ("Kill the process.", "Which process should I terminate?"),
        ("Block off some time.", "For what, and at what date and time?"),
        ("Write it down.", "What should I write, and where should it go?"),
        ("Set up the folder.", "What should the new folder be named?"),
    ],
    "eval": [
        ("Send the report.", "Which report, and who should receive it?"),
        ("Deploy it.", "Which service should I deploy, and to which environment?"),
        ("Translate this.", "What text should I translate, and into which language?"),
        ("Rename the file.", "Which file, and what should the new name be?"),
        ("Cancel my subscription.", "Which subscription would you like me to cancel?"),
        ("Update the record.", "Which record, and what fields should change?"),
        ("Share the doc.", "Which document, and who should I share it with?"),
        ("Restart the service.", "Which service would you like me to restart?"),
        ("Add a label.", "Which item, and what label should I add?"),
        ("Export the data.", "Which dataset, and in what format?"),
    ],
}

# Abstain: NO tool should fire (chit-chat, direct knowledge, tool-ish-looking statements that
# are reports of already-done work). target is the plain assistant reply (no tool, kind="text").
ABSTAIN = {
    "train": [
        ("Thanks, that's really helpful!", "Glad I could help!"),
        ("I already ran the tests, thanks.", "Great, sounds like you're all set."),
        ("Nice, the deploy worked.", "Awesome, glad the deploy went smoothly!"),
        ("How are you doing today?", "I'm doing well, thanks for asking!"),
        ("What can you help me with?", "I can help with coding, research, and everyday tasks."),
        ("I just committed the fix myself.", "Nice work getting that committed."),
        ("Good morning!", "Good morning! How can I help?"),
        ("That makes sense, thank you.", "You're welcome!"),
        ("I'm going to grab a coffee.", "Enjoy your coffee!"),
        ("We already merged the PR.", "Great, glad the PR is merged."),
        ("Perfect, that worked.", "Awesome, glad that did the trick!"),
        ("I already opened the file, no need.", "Got it, sounds like you've got it covered."),
        ("Never mind, I figured it out.", "Nice, glad you sorted it out!"),
        ("You're a lifesaver.", "Happy to help anytime!"),
        ("I closed the ticket on my end.", "Great, thanks for wrapping that up."),
        ("Hey there.", "Hey! What can I do for you?"),
        ("I restarted the server already.", "Good call, sounds like it's back up."),
        ("Appreciate the help earlier.", "Anytime — glad it worked out."),
        ("We pushed the release last night.", "Nice, congrats on getting it out."),
        ("I'm heading out for lunch.", "Enjoy your lunch!"),
    ],
    "eval": [
        ("Have a great weekend!", "Thanks, you too!"),
        ("I finished writing the docs already.", "Nice, glad the docs are done."),
        ("Cool, the build passed on its own.", "Great to hear the build is green."),
        ("Just wanted to say hi.", "Hi there! How can I help?"),
        ("I rolled back the change myself.", "Good call rolling that back."),
        ("Tell me a fun fact.", "Honey never spoils — it can last for thousands of years."),
        ("Thanks a ton for earlier.", "My pleasure — glad it helped!"),
        ("I already cleared the cache.", "Great, that should sort it out."),
        ("Catch you later.", "See you — reach out anytime!"),
        ("We sorted the outage out ourselves.", "Glad to hear it's resolved."),
    ],
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _pool(d: dict, split: str) -> list:
    return d[split]


# ---- single-turn sub-type makers --------------------------------------------------------------
def _clarify(rng: random.Random, split: str) -> Sample:
    prompt, question = rng.choice(CLARIFY[split])
    # kind="text", no tool call: target is the clarifying question itself.
    return Sample("clarify", "clarify", prompt, "text", question)


def _abstain(rng: random.Random, split: str) -> Sample:
    prompt, reply = rng.choice(ABSTAIN[split])
    return Sample("abstain", "abstain", prompt, "text", reply)


def _parallel(rng: random.Random, split: str) -> Sample:
    """A batch request -> >=2 tool calls. Same Sample shape as ``agent_synth.parallel``:
    ``calls`` is the list of dicts, ``target`` the space-joined canonical JSON, ``ref_name`` /
    ``ref_args`` the FIRST call. Every arg value is a literal substring of the user prompt."""
    which = rng.choice(["weather3", "mkdir2", "search2", "read2", "grep2",
                        "edit2", "cmd2", "weather2", "install2", "open2"])
    if which == "weather3":
        cs = rng.sample(_pool(S_CITIES, split), 3)
        prompt = f"What's the weather in {cs[0]}, {cs[1]}, and {cs[2]}?"
        calls = [{"name": "get_weather", "arguments": {"city": c}} for c in cs]
    elif which == "weather2":
        cs = rng.sample(_pool(S_CITIES, split), 2)
        prompt = f"Compare the weather in {cs[0]} and {cs[1]}."
        calls = [{"name": "get_weather", "arguments": {"city": c}} for c in cs]
    elif which == "mkdir2":
        ds = rng.sample(_pool(S_DIRS, split), 2)
        prompt = f"Create the directories {ds[0]} and {ds[1]}."
        calls = [{"name": "make_dir", "arguments": {"path": d}} for d in ds]
    elif which == "search2":
        qs = rng.sample(_pool(S_QUERIES, split), 2)
        prompt = f"Search the web for {qs[0]} and for {qs[1]}."
        calls = [{"name": "web_search", "arguments": {"query": q}} for q in qs]
    elif which == "read2":
        ps = rng.sample(_pool(S_PATHS, split), 2)
        prompt = f"Read {ps[0]} and {ps[1]}."
        calls = [{"name": "read_file", "arguments": {"path": p}} for p in ps]
    elif which == "edit2":
        ps = rng.sample(_pool(S_PATHS, split), 2)
        prompt = f"Edit {ps[0]} and {ps[1]}."
        calls = [{"name": "edit_file", "arguments": {"path": p}} for p in ps]
    elif which == "cmd2":
        cmds = rng.sample(_pool(S_COMMANDS, split), 2)
        prompt = f"Run '{cmds[0]}' and '{cmds[1]}'."
        calls = [{"name": "run_command", "arguments": {"command": c}} for c in cmds]
    elif which == "install2":
        pkgs = rng.sample(_pool(S_PKGS, split), 2)
        prompt = f"Install {pkgs[0]} and {pkgs[1]}."
        calls = [{"name": "install_package", "arguments": {"name": p}} for p in pkgs]
    elif which == "open2":
        us = rng.sample(_pool(S_URLS, split), 2)
        prompt = f"Open {us[0]} and {us[1]}."
        calls = [{"name": "open_url", "arguments": {"url": u}} for u in us]
    else:  # grep2
        pats = rng.sample(_pool(S_PATTERNS, split), 2)
        prompt = f"Search the code for '{pats[0]}' and '{pats[1]}'."
        calls = [{"name": "grep_search", "arguments": {"pattern": p}} for p in pats]
    target = " ".join(_tool_target(c["name"], c["arguments"]) for c in calls)
    ref_args = json.dumps(calls[0]["arguments"], separators=(",", ":"), sort_keys=True)
    return Sample("parallel", "parallel", prompt, "tool", target,
                  calls[0]["name"], ref_args, calls)


_SINGLE_MAKERS = (_clarify, _abstain, _parallel)


def scenario_samples(n: int = 1, seed: int = 0, split: str = "train") -> list[Sample]:
    """Single-turn scenario Samples, ``n`` per sub-type, balanced across {clarify, abstain,
    parallel}. ``split="eval"`` draws DISJOINT phrasing skeletons AND slot values."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = _rng(seed)
    out: list[Sample] = []
    for maker in _SINGLE_MAKERS:
        got: set[str] = set()
        tries = 0
        made = 0
        while made < n and tries < n * 80:
            tries += 1
            s = maker(rng, split)
            if s.prompt in got:
                continue
            got.add(s.prompt)
            out.append(s)
            made += 1
    rng.shuffle(out)
    return out


# ---- multi-turn episode builders --------------------------------------------------------------
def _U(content: str) -> Message:
    return Message(role=Role.user, content=content)


def _A(name: str, args: dict) -> Message:
    return Message(role=Role.assistant, tool_calls=[ToolCall(name, args)])


def _T(resp: str) -> Message:
    return Message(role=Role.tool, tool_response=resp)


def _F(content: str) -> Message:
    return Message(role=Role.assistant, content=content)


# --- workflow: realistic tool chains (coding / research / ops). Args grounded in the user turn or
# a prior tool_response. No injected error. ---
def _ep_workflow_coding(rng: random.Random, split: str) -> list[Message]:
    # read -> grep -> edit -> run_tests -> git_commit. grep returns a path that edit copies.
    path = rng.choice(_pool(S_PATHS, split))
    pat = rng.choice(_pool(S_PATTERNS, split))
    msg = rng.choice(_pool(S_COMMITS, split))
    return [
        _U(f"Fix the bug around '{pat}' in {path} and commit '{msg}'."),
        _A("read_file", {"path": path}),
        _T(f"def {pat}():\n    return None  # looks wrong"),
        _A("grep_search", {"pattern": pat}),
        _T(f"{path}:14:    {pat} appears here"),
        _A("edit_file", {"path": path}),
        _T("edited."),
        _A("run_tests", {}),
        _T("All tests passed."),
        _A("git_commit", {"message": msg}),
        _T("Committed 7c1a2b."),
        _F(f"Fixed '{pat}' in {path} and committed '{msg}'."),
    ]


def _ep_workflow_research(rng: random.Random, split: str) -> list[Message]:
    # web_search -> open_url (url grounded from the search response) -> final summary text.
    query = rng.choice(_pool(S_QUERIES, split))
    url = rng.choice(_pool(S_URLS, split))
    return [
        _U(f"Research {query} and open the best result."),
        _A("web_search", {"query": query}),
        _T(f"1. {url} — a solid overview of {query}"),
        _A("open_url", {"url": url}),
        _T("<page loaded>"),
        _F(f"The best source on {query} is {url}."),
    ]


def _ep_workflow_ops(rng: random.Random, split: str) -> list[Message]:
    # run_command (lint) -> grep -> edit (path grounded from the command response).
    cmd = rng.choice(_pool(S_COMMANDS, split))
    path = rng.choice(_pool(S_PATHS, split))
    pat = rng.choice(_pool(S_PATTERNS, split))
    return [
        _U(f"Run '{cmd}' and fix whatever it flags."),
        _A("run_command", {"command": cmd}),
        _T(f"{path}:22: warning near {pat}"),
        _A("grep_search", {"pattern": pat}),
        _T(f"{path}:22:    {pat}(...)"),
        _A("edit_file", {"path": path}),
        _T("edited."),
        _F(f"Ran '{cmd}' and fixed {path}."),
    ]


def _ep_workflow_review(rng: random.Random, split: str) -> list[Message]:
    # git_status -> git_diff -> run_tests -> git_commit. Pure review-then-commit loop.
    msg = rng.choice(_pool(S_COMMITS, split))
    path = rng.choice(_pool(S_PATHS, split))
    return [
        _U(f"Review my pending changes and, if they look good, commit '{msg}'."),
        _A("git_status", {}),
        _T(f"modified: {path}"),
        _A("git_diff", {}),
        _T(f"@@ {path} @@  - old\n  + new"),
        _A("run_tests", {}),
        _T("All tests passed."),
        _A("git_commit", {"message": msg}),
        _T("Committed 9f3e1d."),
        _F(f"Changes in {path} looked good; committed '{msg}'."),
    ]


def _ep_workflow_setup(rng: random.Random, split: str) -> list[Message]:
    # make_dir -> write_file -> run_command. Scaffolding a new directory then a file in it.
    d = rng.choice(_pool(S_DIRS, split))
    cmd = rng.choice(_pool(S_COMMANDS, split))
    path = rng.choice(_pool(S_PATHS, split))
    return [
        _U(f"Set up {d}, drop {path} in it, then run '{cmd}'."),
        _A("make_dir", {"path": d}),
        _T(f"created {d}."),
        _A("write_file", {"path": path}),
        _T("wrote 18 lines."),
        _A("run_command", {"command": cmd}),
        _T("ok, exit 0."),
        _F(f"Created {d}, wrote {path}, and ran '{cmd}'."),
    ]


def _ep_workflow_research_deep(rng: random.Random, split: str) -> list[Message]:
    # web_search -> open_url -> notion_write (capture the finding). url grounded from search.
    query = rng.choice(_pool(S_QUERIES, split))
    url = rng.choice(_pool(S_URLS, split))
    return [
        _U(f"Look into {query}, open the best link, and save a note about it."),
        _A("web_search", {"query": query}),
        _T(f"Top result: {url} — the canonical guide to {query}"),
        _A("open_url", {"url": url}),
        _T("<page loaded>"),
        _A("notion_write", {"content": query}),
        _T("saved to Notion."),
        _F(f"Read {url} on {query} and saved a note."),
    ]


# --- chained: a tool response carries a value that becomes the next tool's argument. ---
def _ep_chained_grep_edit(rng: random.Random, split: str) -> list[Message]:
    # grep returns a PATH; that exact path becomes edit_file's argument (grounded in tool_response).
    path = rng.choice(_pool(S_PATHS, split))
    pat = rng.choice(_pool(S_PATTERNS, split))
    return [
        _U(f"Find '{pat}' and edit the file it lives in."),
        _A("grep_search", {"pattern": pat}),
        _T(f"{path}:31:    {pat} = build()"),       # path "returned" by grep
        _A("edit_file", {"path": path}),            # grounded from the tool response above
        _T("edited."),
        _F(f"Edited {path} where '{pat}' lives."),
    ]


def _ep_chained_search_open(rng: random.Random, split: str) -> list[Message]:
    # search returns a URL; that exact url becomes open_url's argument (grounded in tool_response).
    query = rng.choice(_pool(S_QUERIES, split))
    url = rng.choice(_pool(S_URLS, split))
    return [
        _U(f"Look up {query} and open the top link."),
        _A("web_search", {"query": query}),
        _T(f"Top hit: {url} (covers {query})"),     # url "returned" by search
        _A("open_url", {"url": url}),               # grounded from the tool response above
        _T("<page loaded>"),
        _F(f"Opened the top result {url} for {query}."),
    ]


def _ep_chained_status_edit(rng: random.Random, split: str) -> list[Message]:
    # git_status names a modified PATH; that exact path becomes edit_file's argument.
    path = rng.choice(_pool(S_PATHS, split))
    return [
        _U("Check what's changed and edit the file that's been touched."),
        _A("git_status", {}),
        _T(f"modified: {path} (unstaged)"),          # path "returned" by status
        _A("edit_file", {"path": path}),             # grounded from the tool response above
        _T("edited."),
        _F(f"Edited the modified file {path}."),
    ]


def _ep_chained_ls_read(rng: random.Random, split: str) -> list[Message]:
    # list_dir returns a PATH inside a dir; that exact path becomes read_file's argument.
    d = rng.choice(_pool(S_DIRS, split))
    path = rng.choice(_pool(S_PATHS, split))
    return [
        _U(f"List the files in {d} and read the main module you find."),
        _A("list_dir", {"path": d}),                 # dir grounded in the user turn
        _T(f"{d}\n{path}\nREADME.md"),               # path "returned" by listing
        _A("read_file", {"path": path}),             # grounded from the tool response above
        _T("<file contents>"),
        _F(f"Read {path} from the listing."),
    ]


# --- error_recovery: a tool turn returns an error / empty result, then the assistant RETRIES with
# corrected args (grounded in that error response) and succeeds. ---
def _ep_error_grep_retry(rng: random.Random, split: str) -> list[Message]:
    # first grep finds nothing; the error names the real symbol -> retry grep with it, then read.
    pat, real = rng.sample(_pool(S_PATTERNS, split), 2)
    path = rng.choice(_pool(S_PATHS, split))
    return [
        _U(f"Find where '{pat}' is defined."),
        _A("grep_search", {"pattern": pat}),
        _T(f"Error: no matches for '{pat}'. Did you mean '{real}'?"),  # error names the real symbol
        _A("grep_search", {"pattern": real}),       # retry, grounded from the error response
        _T(f"{path}:8:    def {real}(): ..."),
        _A("read_file", {"path": path}),             # path grounded from the retry response
        _T("<file contents>"),
        _F(f"'{pat}' was actually '{real}', defined in {path}."),
    ]


def _ep_error_read_retry(rng: random.Random, split: str) -> list[Message]:
    # first read hits a missing file; the error gives the correct path -> retry read, then edit.
    wrong = rng.choice(_pool(S_PATHS, split))
    right = rng.choice([p for p in _pool(S_PATHS, split) if p != wrong])
    return [
        _U(f"Open {wrong} and clean it up."),
        _A("read_file", {"path": wrong}),
        _T(f"Error: {wrong} not found. Try {right} instead."),  # error names the correct path
        _A("read_file", {"path": right}),            # retry, grounded from the error response
        _T("<file contents>"),
        _A("edit_file", {"path": right}),            # grounded from the error response
        _T("edited."),
        _F(f"{wrong} didn't exist; cleaned up {right} instead."),
    ]


def _ep_error_cmd_retry(rng: random.Random, split: str) -> list[Message]:
    # a command fails (empty/error); the error suggests another command -> retry succeeds.
    bad = rng.choice(_pool(S_COMMANDS, split))
    good = rng.choice([c for c in _pool(S_COMMANDS, split) if c != bad])
    return [
        _U(f"Run '{bad}' to validate the build."),
        _A("run_command", {"command": bad}),
        _T(f"Error: '{bad}' failed (exit 1). Hint: run '{good}' first."),  # names the retry command
        _A("run_command", {"command": good}),        # retry, grounded from the error response
        _T("ok, exit 0."),
        _F(f"'{bad}' failed, so I ran '{good}' which succeeded."),
    ]


def _ep_error_find_retry(rng: random.Random, split: str) -> list[Message]:
    # find_files with the wrong glob returns nothing; the error suggests the right glob -> retry.
    wrong = rng.choice(_pool(S_PATHS, split))
    right = rng.choice([p for p in _pool(S_PATHS, split) if p != wrong])
    return [
        _U(f"Open {wrong} and summarize it."),
        _A("read_file", {"path": wrong}),
        _T(f"Error: cannot open {wrong} (no such file). Closest match: {right}."),
        _A("read_file", {"path": right}),            # retry, grounded from the error response
        _T("<file contents>"),
        _F(f"{wrong} was missing; summarized {right} instead."),
    ]


def _ep_error_url_retry(rng: random.Random, split: str) -> list[Message]:
    # an open_url 404s; the error gives the correct url -> retry open_url succeeds.
    bad = rng.choice(_pool(S_URLS, split))
    good = rng.choice([u for u in _pool(S_URLS, split) if u != bad])
    return [
        _U(f"Open {bad} and read it."),
        _A("open_url", {"url": bad}),
        _T(f"Error: {bad} returned 404. The page moved to {good}."),  # names the correct url
        _A("open_url", {"url": good}),               # retry, grounded from the error response
        _T("<page loaded>"),
        _F(f"{bad} was a dead link; opened {good} instead."),
    ]


_WORKFLOW = (_ep_workflow_coding, _ep_workflow_research, _ep_workflow_ops,
             _ep_workflow_review, _ep_workflow_setup, _ep_workflow_research_deep)
_CHAINED = (_ep_chained_grep_edit, _ep_chained_search_open,
            _ep_chained_status_edit, _ep_chained_ls_read)
_ERROR = (_ep_error_grep_retry, _ep_error_read_retry, _ep_error_cmd_retry,
          _ep_error_find_retry, _ep_error_url_retry)

_FAMILIES = {
    "workflow": _WORKFLOW,
    "chained": _CHAINED,
    "error_recovery": _ERROR,
}


def scenario_episodes(n: int = 1, seed: int = 0, split: str = "train") -> list[Conversation]:
    """Multi-turn scenario episodes, ``n`` per family, balanced across {workflow, chained,
    error_recovery}. Each is a valid alternating role sequence ending on a final assistant turn.
    ``split="eval"`` draws DISJOINT phrasing skeletons AND slot values from ``split="train"``."""
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = _rng(seed)
    out: list[Conversation] = []
    for family, builders in _FAMILIES.items():
        for _ in range(n):
            builder = rng.choice(builders)
            msgs = builder(rng, split)
            out.append(Conversation(
                messages=msgs,
                meta={"kind": "scenario_episode", "category": family,
                      "type": builder.__name__.removeprefix("_ep_")},
            ))
    rng.shuffle(out)
    return out
