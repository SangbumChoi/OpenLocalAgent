"""Tool-calling across real-world surface forms: MCP servers, REST APIs, CLI commands, SDK calls.

Same machinery as the other benches, but each tool is tagged with a **modality** so we can ask:
does grounding/selection differ by surface form? Args span the realistic mix of each modality —
MCP/REST tend to have named JSON args (often quoted/typed); CLI/SDK lean on positionals, flags,
and bare identifiers (host, image, bucket, instance id), which are harder to ground.
"""

from __future__ import annotations

from localagent.eval.toolcall_bench import build_tools as _build
from localagent.eval.toolcall_bench import examples as _examples
from localagent.eval.toolcall_bench import gold_set as _gold

Q = {"type": "string", "format": "quoted"}
P = {"type": "string", "format": "path"}
U = {"type": "string", "format": "url"}
S = {"type": "string"}
N = {"type": "integer"}

# (modality, name, desc, [(arg, schema, train, eval)], [templates], verb, [syn])
SCENARIOS = [
    # --- MCP servers (namespaced tools, named JSON args) ---
    ("MCP", "mcp__github__create_issue", "create a GitHub issue via the github MCP server",
     [("title", Q, ["flaky test", "login bug"], ["broken link", "slow query"])],
     ["{verb} a GitHub issue '{a0}'.", "{verb} an issue '{a0}' on GitHub."], "create", ["open", "file"]),
    ("MCP", "mcp__filesystem__read_file", "read a file via the filesystem MCP server",
     [("path", P, ["src/a.py", "docs/x.md"], ["api/y.go", "web/z.ts"])],
     ["{verb} the file {a0}.", "{verb} {a0}."], "read", ["open", "show"]),
    ("MCP", "mcp__slack__post_message", "post a Slack message via the slack MCP server",
     [("text", Q, ["deploy done", "standup in 5"], ["ship it", "needs review"])],
     ["{verb} '{a0}' to Slack.", "{verb} a Slack message '{a0}'."], "post", ["send", "message"]),
    ("MCP", "mcp__fetch__get_url", "fetch a URL via the fetch MCP server",
     [("url", U, ["example.com", "python.org"], ["figma.com", "openai.com"])],
     ["{verb} {a0}.", "{verb} the page {a0}."], "fetch", ["get", "grab"]),
    ("MCP", "mcp__memory__store", "store a memory via the memory MCP server",
     [("content", Q, ["the api key", "user prefers dark mode"], ["the build path", "owner is Sam"])],
     ["{verb} '{a0}' in memory.", "{verb} that '{a0}'."], "remember", ["store", "save"]),
    ("MCP", "mcp__sqlite__query", "run a SQL query via the sqlite MCP server",
     [("sql", Q, ["select * from users", "count orders"], ["drop temp", "list tables"])],
     ["{verb} the query '{a0}'.", "{verb} '{a0}'."], "run", ["execute", "exec"]),

    # --- REST APIs (endpoint + params) ---
    ("REST", "create_user", "POST /users — create a user",
     [("name", S, ["Alice", "Bob"], ["Greta", "Mateo"])],
     ["{verb} a user named {a0}.", "{verb} the user {a0}."], "create", ["add", "register"]),
    ("REST", "get_order", "GET /orders/{id} — fetch an order",
     [("order_id", Q, ["A100", "B200"], ["C300", "D400"])],
     ["{verb} order '{a0}'.", "{verb} the order '{a0}'."], "get", ["fetch", "look up"]),
    ("REST", "update_status", "PATCH /tickets — set a ticket status",
     [("status", {"type": "string", "enum": ["open", "closed", "pending"]}, ["open", "closed"], ["pending", "closed"])],
     ["{verb} the status to {a0}.", "{verb} it to {a0}."], "set", ["change", "mark"]),
    ("REST", "search_products", "GET /products?q= — search products",
     [("query", S, ["running shoes", "blue mug"], ["wool socks", "desk lamp"])],
     ["{verb} products for {a0}.", "{verb} for {a0} in products."], "search", ["find", "look"]),
    ("REST", "post_comment", "POST /comments — add a comment",
     [("text", Q, ["nice work", "please fix"], ["lgtm", "needs tests"])],
     ["{verb} a comment '{a0}'.", "{verb} '{a0}' as a comment."], "post", ["add", "leave"]),
    ("REST", "delete_session", "DELETE /sessions/{token}",
     [("token", Q, ["abc123", "def456"], ["xyz789", "uvw000"])],
     ["{verb} session '{a0}'.", "{verb} the session '{a0}'."], "delete", ["end", "kill"]),

    # --- CLI commands (positionals + flags) ---
    ("CLI", "docker_run", "docker run an image on a port",
     [("image", S, ["nginx", "redis"], ["postgres", "mongo"]), ("port", N, ["8080", "5000"], ["3000", "9090"])],
     ["{verb} the {a0} container on port {a1}.", "{verb} {a0} on port {a1}."], "run", ["start", "launch"]),
    ("CLI", "kubectl_get", "kubectl get a resource in a namespace",
     [("resource", S, ["pods", "services"], ["deployments", "nodes"]),
      ("namespace", Q, ["default", "kube-system"], ["staging", "prod"])],
     ["{verb} the {a0} in namespace '{a1}'.", "{verb} {a0} in '{a1}'."], "get", ["list", "show"]),
    ("CLI", "git_clone", "git clone a repository",
     [("repo", U, ["github.com/a/b", "gitlab.com/x/y"], ["github.com/p/q", "bitbucket.org/m/n"])],
     ["{verb} {a0}.", "{verb} the repo {a0}."], "clone", ["clone", "checkout"]),
    ("CLI", "npm_install", "npm install a package",
     [("package", Q, ["express", "react"], ["lodash", "axios"])],
     ["{verb} the package '{a0}'.", "{verb} '{a0}'."], "install", ["add", "get"]),
    ("CLI", "ssh_connect", "ssh into a host",
     [("host", U, ["server.example.com", "db.internal.net"], ["app.prod.io", "cache.dev.org"])],
     ["{verb} into {a0}.", "{verb} to {a0}."], "ssh", ["connect", "log in"]),
    ("CLI", "tar_extract", "extract a tar archive",
     [("archive", P, ["backup.tar.gz", "data.zip"], ["logs.tar", "dump.tgz"])],
     ["{verb} {a0}.", "{verb} the archive {a0}."], "extract", ["unpack", "untar"]),

    # --- SDK calls (library method invocations, identifiers) ---
    ("SDK", "s3_upload", "boto3 S3 upload_file to a key",
     [("key", P, ["reports/q3.pdf", "data/x.csv"], ["logs/y.txt", "img/z.png"])],
     ["{verb} to S3 key {a0}.", "{verb} the file to {a0} on S3."], "upload", ["put", "push"]),
    ("SDK", "openai_chat", "openai client chat with a prompt",
     [("prompt", Q, ["summarize this", "write a haiku"], ["explain recursion", "draft an email"])],
     ["{verb} the model with prompt '{a0}'.", "{verb} '{a0}' to the model."], "call", ["ask", "prompt"]),
    ("SDK", "redis_set", "redis client SET a value",
     [("value", Q, ["enabled", "42"], ["dark", "ready"])],
     ["{verb} the redis value to '{a0}'.", "{verb} redis to '{a0}'."], "set", ["set", "update"]),
    ("SDK", "boto3_start_instance", "boto3 EC2 start_instances",
     [("instance_id", Q, ["i-abc123", "i-def456"], ["i-xyz789", "i-uvw000"])],
     ["{verb} instance '{a0}'.", "{verb} the EC2 instance '{a0}'."], "start", ["boot", "launch"]),
    ("SDK", "pandas_read_csv", "pandas read_csv a path",
     [("path", P, ["data/train.csv", "out/x.csv"], ["raw/y.csv", "tmp/z.csv"])],
     ["{verb} the CSV {a0}.", "{verb} {a0} with pandas."], "read", ["load", "open"]),
    ("SDK", "stripe_charge", "stripe Charge.create an amount",
     [("amount", N, ["100", "50"], ["250", "75"]),
      ("currency", {"type": "string", "enum": ["usd", "eur", "gbp"]}, ["usd", "eur"], ["gbp", "usd"])],
     ["{verb} {a0} {a1}.", "{verb} the card {a0} {a1}."], "charge", ["bill", "charge"]),
]

MODALITY = {t[1]: t[0] for t in SCENARIOS}
_DEFS = [t[1:] for t in SCENARIOS]            # drop modality for the shared machinery


def build_tools():
    return _build(_DEFS)


def examples():
    return _examples(_DEFS)


def gold_set(split="eval", seed=0):
    return _gold(split, seed, _DEFS)


IRRELEVANT = ["Tell me a joke.", "How are you?", "I love you.", "Sing a song.",
              "What's your name?", "asdf qwer", "Good morning.", "Thanks a lot."]
