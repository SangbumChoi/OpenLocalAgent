"""A software/coding tool-calling benchmark (SWE-agent-style tools).

~24 realistic developer tools with real JSON schemas — files, search, VCS, build/test, code-intel,
PRs — many **multi-argument** (e.g. ``edit_file(path, old_string, new_string)`` is 3 args). Stresses
argument grounding far harder than the general suite. Same machinery as ``toolcall_bench`` (disjoint
train/eval slot values, paraphrased eval verbs, irrelevant queries for abstention).
"""

from __future__ import annotations

from localagent.eval.toolcall_bench import build_tools as _build
from localagent.eval.toolcall_bench import examples as _examples
from localagent.eval.toolcall_bench import gold_set as _gold

# name, desc, [(arg, schema, train_vals, eval_vals)], [templates], verb, [synonyms]
P = {"type": "string", "format": "path"}
Q = {"type": "string", "format": "quoted"}
S = {"type": "string"}
INT_SCHEMA = {"type": "integer"}
_PATHS_T, _PATHS_E = ["src/app.py", "lib/util.js"], ["api/routes.go", "web/main.ts"]
_PATHS_T2, _PATHS_E2 = ["test/old.py", "core/a.py"], ["pkg/x.rs", "cmd/run.go"]

CODE_TOOLS = [
    ("read_file", "read the contents of a file",
     [("path", P, _PATHS_T, _PATHS_E)],
     ["{verb} {a0}.", "{verb} the file {a0}."], "read", ["open", "show", "cat"]),
    ("write_file", "create or overwrite a file",
     [("path", P, _PATHS_T, _PATHS_E)],
     ["{verb} {a0}.", "{verb} the file {a0}."], "create", ["write", "make"]),
    ("edit_file", "replace text in a file",
     [("path", P, _PATHS_T, _PATHS_E),
      ("old_string", Q, ["foo", "TODO"], ["bar", "FIXME"]),
      ("new_string", Q, ["baz", "done"], ["qux", "fixed"])],
     ["{verb} '{a1}' with '{a2}' in {a0}.", "In {a0}, {verb} '{a1}' with '{a2}'."],
     "replace", ["swap", "change"]),
    ("delete_file", "delete a file",
     [("path", P, ["tmp/x.log", "build/y.o"], ["out/z.bin", "cache/w.tmp"])],
     ["{verb} {a0}.", "{verb} the file {a0}."], "delete", ["remove", "rm"]),
    ("move_file", "move or rename a file",
     [("source", P, _PATHS_T2, _PATHS_E2), ("dest", P, ["arch/o.py", "bak/a.py"], ["old/x.rs", "tmp/r.go"])],
     ["{verb} {a0} to {a1}.", "{verb} the file {a0} to {a1}."], "move", ["rename", "relocate"]),
    ("create_directory", "create a directory",
     [("path", P, ["src/utils", "tests/unit"], ["pkg/api", "web/views"])],
     ["{verb} the directory {a0}.", "{verb} a folder {a0}."], "create", ["make", "add"]),
    ("list_directory", "list the files in a directory",
     [("path", P, ["src", "tests"], ["pkg", "web"])],
     ["{verb} the directory {a0}.", "{verb} the files in {a0}."], "list", ["show", "ls"]),
    ("grep_search", "search the codebase for a regex pattern",
     [("pattern", Q, ["def main", "import os"], ["class User", "async fn"])],
     ["{verb} for '{a0}'.", "{verb} the code for '{a0}'."], "grep", ["search", "find"]),
    ("find_symbol", "find where a symbol is defined",
     [("symbol", Q, ["parse_args", "Config"], ["handle_request", "Logger"])],
     ["{verb} the symbol '{a0}'.", "Where is '{a0}' {verb}?"], "defined", ["located", "declared"]),
    ("glob_files", "list files matching a glob pattern",
     [("pattern", Q, ["*.py", "src/**/*.js"], ["**/*.go", "test_*.py"])],
     ["{verb} files matching '{a0}'.", "{verb} '{a0}'."], "find", ["glob", "list"]),
    ("git_commit", "create a git commit",
     [("message", Q, ["fix bug", "add tests"], ["tidy imports", "bump deps"])],
     ["{verb} with message '{a0}'.", "{verb} '{a0}'."], "commit", ["check in", "save"]),
    ("git_checkout", "switch to a git branch",
     [("branch", Q, ["main", "dev"], ["release", "hotfix"])],
     ["{verb} the branch '{a0}'.", "{verb} to '{a0}'."], "checkout", ["switch", "go"]),
    ("git_revert", "revert a git commit",
     [("commit", S, ["abc123", "def456"], ["aa11bb", "cc22dd"])],
     ["{verb} commit {a0}.", "{verb} the commit {a0}."], "revert", ["undo", "roll back"]),
    ("run_tests", "run the test suite",
     [("path", P, ["tests/test_a.py", "test/b.js"], ["pkg/c_test.go", "spec/d.ts"])],
     ["{verb} the tests in {a0}.", "{verb} {a0}."], "run", ["execute", "exec"]),
    ("run_command", "run a shell command",
     [("command", Q, ["ls -la", "npm ci"], ["go build", "cargo run"])],
     ["{verb} '{a0}'.", "{verb} the command '{a0}'."], "run", ["execute", "exec"]),
    ("lint_file", "lint a file",
     [("path", P, ["src/api.py", "lib/db.js"], ["pkg/v.go", "web/u.ts"])],
     ["{verb} {a0}.", "{verb} the file {a0}."], "lint", ["check", "lint"]),
    ("format_file", "auto-format a file",
     [("path", P, ["src/m.py", "lib/n.js"], ["pkg/o.go", "web/p.ts"])],
     ["{verb} {a0}.", "{verb} the file {a0}."], "format", ["reformat", "prettify"]),
    ("install_package", "install a dependency",
     [("package", Q, ["numpy", "express"], ["torch", "axios"])],
     ["{verb} the package '{a0}'.", "{verb} '{a0}'."], "install", ["add", "get"]),
    ("rename_symbol", "rename a symbol across the codebase",
     [("old_name", Q, ["getUser", "calc"], ["doThing", "tmp"]),
      ("new_name", Q, ["fetchUser", "compute"], ["execute", "buffer"])],
     ["{verb} '{a0}' to '{a1}'.", "{verb} the symbol '{a0}' to '{a1}'."], "rename", ["change", "refactor"]),
    ("add_comment", "add a comment to a file",
     [("path", P, ["src/h.py", "lib/i.js"], ["pkg/j.go", "web/k.ts"]),
      ("comment", Q, ["needs review", "TODO refactor"], ["fix later", "deprecated"])],
     ["{verb} a comment '{a1}' to {a0}.", "{verb} '{a1}' to {a0}."], "add", ["insert", "put"]),
    ("create_pull_request", "open a pull request",
     [("title", Q, ["add caching", "fix login"], ["update docs", "drop dead code"])],
     ["{verb} a pull request '{a0}'.", "{verb} a PR titled '{a0}'."], "open", ["create", "file"]),
    ("comment_on_pr", "comment on a pull request",
     [("number", INT_SCHEMA, ["42", "7"], ["13", "99"]),
      ("body", Q, ["looks good", "needs work"], ["ship it", "nit"])],
     ["{verb} on PR {a0} with '{a1}'.", "{verb} '{a1}' on PR {a0}."], "comment", ["reply", "note"]),
    ("create_issue", "create an issue",
     [("title", Q, ["flaky test", "memory leak"], ["broken link", "slow query"])],
     ["{verb} an issue '{a0}'.", "{verb} a bug report '{a0}'."], "create", ["file", "open"]),
    ("get_definition", "get the definition of a symbol",
     [("symbol", Q, ["Router", "to_json"], ["Cache", "from_dict"])],
     ["{verb} the definition of '{a0}'.", "{verb} '{a0}'."], "show", ["get", "find"]),
]


def build_tools():
    return _build(CODE_TOOLS)


def examples():
    return _examples(CODE_TOOLS)


def gold_set(split="eval", seed=0):
    return _gold(split, seed, CODE_TOOLS)


IRRELEVANT = ["Tell me a joke.", "What's the weather?", "How are you?", "Sing a song.",
              "I love you.", "asdf qwer", "Thanks!", "Who are you?", "Good morning."]
