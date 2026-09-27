"""Paraphrase-rich synthetic tool-calling data (intent -> tool generalization).

Why this module exists: the templated ``agent_synth.Generator`` uses a *narrow* band of
phrasings per tool, so the selector overfits surface wording and mis-routes natural,
out-of-distribution queries (e.g. "What is the color of a monkey?" -> get_news instead of
web_search; "Search the codebase for train_step" -> define instead of grep_search). This source
gives MANY varied natural phrasings per tool — imperative, question, and colloquial forms with
synonyms — so the model learns intent -> tool mapping that generalizes.

Contracts honored (other code depends on these):
  * Reuses the existing ``Sample`` dataclass and ``_tool_target`` canonical-JSON helper from
    ``agent_synth`` — schema/config untouched.
  * Every argument value is a LITERAL substring of ``prompt`` (the pointer-copy head copies args
    verbatim). For quoted-format tools the value is single-quoted inside the phrasing.
  * Train vs eval are DISJOINT in BOTH phrasing templates AND slot values, so ``split="eval"``
    truly measures generalization rather than memorization.

Public API (consumed by retrain code):
  * ``paraphrase_samples(n=1, seed=0, split="train") -> list[Sample]`` — balanced across all 50
    tools in ``STANDARD_TOOLS``; ``n`` is the number of samples PER tool.
  * ``TOOL_EXAMPLES: dict[str, list[str]]`` — representative train-pool phrasings per tool, used
    to seed example-query tool embeddings for the retriever/selector.
"""

from __future__ import annotations

import json
import random

from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.agent_synth import Sample, _tool_target

# ---------------------------------------------------------------------------------------------
# Slot pools — DISJOINT train/eval. These are paraphrase-local (kept here, not imported, so the
# eval pool here never overlaps the train pool here). Values are simple, single-token-ish so they
# ground as literal substrings under any phrasing template.
# ---------------------------------------------------------------------------------------------

# free-form web_search queries: bare facts/questions a person actually types
WEB_QUERIES = {
    "train": ["the color of a monkey", "the tallest mountain", "why the sky is blue",
              "the speed of sound", "how old the universe is", "the largest desert",
              "what causes thunder", "the boiling point of mercury", "the population of Brazil",
              "the fastest car", "how bees make honey", "the deepest lake",
              "the height of a giraffe", "the lifespan of a turtle", "what a black hole is",
              "the distance to the moon", "the smallest country", "how earthquakes happen",
              "the weight of an elephant", "the oldest tree", "the brightest star",
              "the longest bridge", "what causes rainbows", "the diet of a panda"],
    "eval": ["the color of a flamingo", "the widest river", "why leaves change color",
             "the speed of a cheetah", "how glaciers form", "the largest ocean",
             "what causes tides", "the freezing point of alcohol", "the population of Japan",
             "the heaviest animal", "how spiders spin webs", "the highest waterfall"],
}
# define: single words / short terms (meaning questions)
DEFINE_TERMS = {
    "train": ["serendipity", "ephemeral", "quintessential", "ubiquitous", "altruism",
              "paradox", "nostalgia", "resilience", "ambiguous", "pragmatic", "eloquent",
              "meticulous", "candor", "juxtaposition", "tenacity", "empathy", "obsolete",
              "verbose", "pristine", "lucid", "frugal", "diligent", "candid", "vivid"],
    "eval": ["catharsis", "esoteric", "gregarious", "panacea", "zealous", "austere",
             "benevolent", "cryptic", "fervent", "languid"],
}
# get_news: current-event topics
NEWS_TOPICS = {
    "train": ["the latest tech layoffs", "the election results", "the new tax bill",
              "the championship game", "the interest rate decision", "the climate summit",
              "the new iPhone launch", "the merger talks", "the protests downtown",
              "the space launch", "the celebrity scandal", "the trade negotiations"],
    "eval": ["the central bank meeting", "the wildfire updates", "the peace talks",
             "the box office numbers", "the vaccine rollout", "the budget deal"],
}
# open_url / http_request: domains (a homepage / endpoint is fine for these two).
DOMAINS = {
    "train": ["acme.io", "store.example.net", "blog.dev", "news.site.org", "shop.co",
              "portal.app", "files.example.com", "cdn.assets.io", "api.service.dev",
              "docs.product.com", "status.host.io", "mail.provider.net"],
    "eval": ["forum.example.org", "media.host.com", "data.service.io", "wiki.project.net",
             "downloads.app.co", "support.vendor.com"],
}
# download_file: ALWAYS a concrete file URL (path + extension), never a bare domain/status page.
# Disjoint train/eval. Each value is a literal substring of the prompt.
DOWNLOAD_URLS = {
    "train": ["https://cdn.acme.io/release.zip", "files.example.com/report.pdf",
              "https://assets.site.org/logo.png", "downloads.host.io/tool.deb",
              "https://blog.dev/assets/data.csv", "cdn.assets.io/images.tar.gz",
              "https://docs.product.com/manual.pdf", "store.example.net/export.json",
              "https://portal.app/files/backup.zip", "mirror.service.dev/pkg.whl",
              "https://files.example.com/sample.xlsx", "cdn.acme.io/fonts.woff2"],
    "eval": ["https://media.host.com/clip.mp4", "downloads.app.co/installer.dmg",
             "https://data.service.io/dump.sql", "wiki.project.net/files/notes.txt",
             "https://support.vendor.com/patch.tar.gz", "forum.example.org/upload.png"],
}
# list_dir / find_files (glob) / make_dir / read_file / write_file / edit_file: paths & names
DIR_PATHS = {
    "train": ["src", "tests", "docs", "config", "scripts", "assets", "vendor", "logs",
              "examples", "templates", "migrations", "fixtures"],
    "eval": ["build", "dist", "output", "cache", "tmp", "backups"],
}
FILE_PATHS = {
    "train": ["notes.md", "main.go", "index.ts", "setup.cfg", "Makefile", "report.csv",
              "schema.graphql", "server.rb", "handler.kt", "styles.css", "queries.sql",
              "pipeline.yml"],
    "eval": ["changelog.txt", "app.swift", "model.proto", "deps.lock", "routes.rb",
             "view.vue"],
}
GLOB_PATTERNS = {
    "train": ["*.json", "**/*.md", "test_*.go", "*.yml", "src/**/*.rb", "*.proto",
              "*.cfg", "**/*.kt", "*.graphql", "*.lock"],
    "eval": ["*.ini", "**/*.swift", "*.vue", "spec/**/*.rb", "*.tf", "*.bat"],
}
# grep_search: code symbols / patterns people actually search for
GREP_PATTERNS = {
    "train": ["train_step", "load_config", "def parse", "TODO", "class Server",
              "handle_request", "import torch", "API_TOKEN", "retry_count", "on_click",
              "validate_input", "build_query"],
    "eval": ["run_epoch", "save_checkpoint", "class Router", "FIXME", "fetch_user",
             "parse_args"],
}
# run_command: shell commands
SHELL_COMMANDS = {
    "train": ["ls -la", "df -h", "ps aux", "top", "whoami", "uname -a", "make build",
              "npm ci", "cargo test", "go vet ./...", "rake db:migrate", "tox"],
    "eval": ["du -sh", "free -m", "make clean", "yarn lint", "cargo fmt", "bundle install"],
}
# run_python: ACTUAL inline Python code snippets (never a bare script filename — running a
# `.py` file is a shell command and lives in run_command). Values ground as literal substrings.
PY_SNIPPETS = {
    "train": ["sum([1, 2, 3])", "print(42)", "len('hello')", "max(3, 9)", "abs(-7)",
              "round(3.14159, 2)", "sorted([3, 1, 2])", "range(5)", "int('100')",
              "str.upper('hi')", "bool(0)", "list(reversed([1, 2]))",
              "print(sum(range(10)))", "[x*x for x in range(4)]"],
    "eval": ["min(8, 2)", "pow(2, 10)", "divmod(17, 5)", "any([0, 1])", "tuple([1, 2])",
             "hex(255)", "print(len([1, 2, 3]))"],
}
# send_email: recipients
EMAIL_NAMES = {
    "train": ["Marco", "Sofia", "Liam", "Aisha", "Tariq", "Elena", "Kenji", "Priya",
              "Diego", "Maya", "Omar", "Freya"],
    "eval": ["Lucas", "Nadia", "Theo", "Amara", "Felix", "Yusuf"],
}
# calendar_event: titles
CAL_TITLES = {
    "train": ["dentist appointment", "team lunch", "1:1 with manager", "gym session",
              "product launch", "client call", "yoga class", "book club", "haircut",
              "parent meeting", "flight to NYC", "vaccine shot"],
    "eval": ["coffee with Sam", "tax deadline", "guitar lesson", "house viewing",
             "doctor visit", "company offsite"],
}
# slack_send: messages
SLACK_MSGS = {
    "train": ["running late", "PR is up for review", "deploy finished", "joining now",
              "can someone take a look", "build is broken", "out for lunch", "fixed it",
              "great work everyone", "standup in 10", "merging this", "lgtm"],
    "eval": ["heading home", "tests passing now", "need a quick review", "back from break",
             "on a call", "shipping today"],
}
# install_package: package names
PKG_NAMES = {
    "train": ["lodash", "axios", "express", "vite", "eslint", "jest", "webpack", "babel",
              "tailwind", "prettier", "vitest", "zod"],
    "eval": ["rollup", "vue", "svelte", "nextjs", "prisma", "drizzle"],
}
# kill_process: process names
PROC_NAMES = {
    "train": ["node", "ruby", "php", "dotnet", "deno", "bun", "esbuild", "watchman",
              "ngrok", "caddy", "envoy", "consul"],
    "eval": ["vault", "nats", "etcd", "minio", "loki", "tempo"],
}


def _pool(pool: dict, split: str) -> list:
    return pool["train"] if split == "train" else pool["eval"]


# ---------------------------------------------------------------------------------------------
# Phrasing templates — DISJOINT train/eval. Each entry is a (templates, group, name, arg, pool,
# quoted) spec. ``{v}`` is the slot. ``quoted=True`` wraps the value in single quotes so quoted-
# format args ground (and stay literal substrings either way). For arg-less tools, ``arg`` is
# ``None`` and the templates carry no ``{v}``.
#
# The big-variety tools (the OOD failures) get many synonyms across imperative / question /
# colloquial registers. Train and eval template lists never share a string.
# ---------------------------------------------------------------------------------------------

# Each value: {"group", "name"(optional, defaults to key), "arg", "pool", "quoted",
#              "train": [...templates...], "eval": [...templates...]}
TOOL_SPECS: dict[str, dict] = {
    # --- big-variety OOD failures -----------------------------------------------------------
    "web_search": {
        "group": "web_search", "arg": "query", "pool": WEB_QUERIES, "quoted": False,
        "train": [
            "What is {v}?", "What's {v}?", "Look up {v}.", "Look up {v} for me.",
            "Google {v}.", "Find out {v}.", "Can you find out {v}?",
            "I want to know {v}.", "Tell me {v}.", "Do you know {v}?",
            "Any idea about {v}?", "Help me figure out {v}.", "Search online for {v}.",
            "Let me know {v}.", "I'm curious about {v}.", "Search the web for {v}.",
            "Hey, what is {v}?", "Remind me, what is {v}?", "I was wondering {v}.",
            "Got any info on {v}?", "I'd like to know {v}.", "Can you tell me {v}?",
            "What's the deal with {v}?", "Mind looking up {v}?",
        ],
        "eval": [
            "Could you look up {v}?", "Find me {v} on the web.", "What would {v} be?",
            "I need to know {v}.", "Go search for {v}.", "Quick question: {v}?",
            "Dig up {v}.", "What in the world is {v}?",
        ],
    },
    "define": {
        "group": "define", "arg": "term", "pool": DEFINE_TERMS, "quoted": False,
        "train": [
            "What does {v} mean?", "Define {v}.", "What's the meaning of {v}?",
            "Explain the word {v}.", "Give me the definition of {v}.",
            "What is {v}?", "Meaning of {v}?", "Can you define {v}?",
            "I don't know the word {v}.", "What does the term {v} mean?",
            "Describe what {v} means.", "Help me understand {v}.",
            "Define the term {v}.", "What's a {v}?", "Look up the meaning of {v}.",
            "What's the definition of the word {v}?", "Spell out what {v} means.",
            "I keep seeing the word {v}; what is it?",
        ],
        "eval": [
            "Define the word {v} for me.", "What's {v} supposed to mean?",
            "Tell me the definition of {v}.", "Explain {v}.",
            "What does {v} refer to?", "Clarify the meaning of {v}.",
            "Break down the word {v} for me.", "What exactly is {v}?",
        ],
    },
    "open_url": {
        "group": "browser", "arg": "url", "pool": DOMAINS, "quoted": False,
        "train": [
            "Open {v}.", "Go to {v}.", "Navigate to {v}.", "Pull up {v}.",
            "Visit {v}.", "Open {v} in the browser.", "Take me to {v}.",
            "Load {v}.", "Bring up {v}.", "Open up {v} please.",
            "Hop over to {v}.", "Fire up {v} in the browser.", "Jump to {v}.",
            "Can you visit {v}?", "Surf to {v}.", "Open the site {v}.",
        ],
        "eval": [
            "Could you open {v}?", "Head over to {v}.", "Browse to {v}.",
            "Show me {v}.", "Let's go to {v}.", "Open the page {v}.",
            "Point the browser at {v}.", "Take a look at {v}.",
        ],
    },
    "http_request": {
        "group": "code", "arg": "url", "pool": DOMAINS, "quoted": False,
        "train": [
            "Make an HTTP request to {v}.", "Hit the endpoint {v}.", "Call the API at {v}.",
            "Send a request to {v}.", "Fetch {v} via HTTP.", "Ping the URL {v}.",
            "Query the endpoint {v}.", "Make a GET request to {v}.",
            "Do a GET on {v}.", "Issue an HTTP request to {v}.", "Hit {v} with a request.",
            "Send an HTTP GET to {v}.", "Call out to {v}.", "Fire an HTTP request at {v}.",
        ],
        "eval": [
            "Do an HTTP call to {v}.", "Request the resource at {v}.",
            "Reach out to the API {v}.", "Send an API call to {v}.",
            "Curl {v}.", "Make a request against {v}.",
            "Perform an HTTP request to {v}.", "Poll the endpoint {v}.",
        ],
    },
    "download_file": {
        "group": "code", "arg": "url", "pool": DOWNLOAD_URLS, "quoted": False,
        "train": [
            "Download the file from {v}.", "Download {v}.", "Grab the file at {v}.",
            "Fetch the file from {v}.", "Save the file from {v}.", "Pull down {v}.",
            "Get the file at {v}.", "Retrieve the file from {v}.",
            "Download the file located at {v}.", "Save {v} to disk.",
            "Wget {v}.", "Pull the file down from {v}.", "Grab {v} for me.",
            "Download whatever is at {v}.",
        ],
        "eval": [
            "Could you download {v}?", "Snag the file from {v}.",
            "Download the asset at {v}.", "Get me the file from {v}.",
            "Fetch {v} and save it.", "Pull the file off {v}.",
            "Save down the file at {v}.", "Fetch the download from {v}.",
        ],
    },
    "get_news": {
        "group": "news", "arg": "topic", "pool": NEWS_TOPICS, "quoted": False,
        "train": [
            "What's the latest on {v}?", "Any news about {v}?", "Catch me up on {v}.",
            "What's happening with {v}?", "Give me the news on {v}.",
            "Any updates on {v}?", "What's new with {v}?", "Show me headlines about {v}.",
            "Brief me on {v}.", "What's going on with {v}?",
            "Pull up the news on {v}.", "Tell me the latest about {v}.",
            "Show me the news for {v}.", "What are people saying about {v}?",
            "Update me on {v}.", "Read me the news on {v}.",
        ],
        "eval": [
            "Fill me in on {v}.", "What's the buzz about {v}?",
            "Latest headlines on {v}?", "News update on {v} please.",
            "Anything new on {v}?", "Get me current news about {v}.",
            "What's the story with {v}?", "Bring me up to speed on {v}.",
        ],
    },
    "list_dir": {
        "group": "code", "arg": "path", "pool": DIR_PATHS, "quoted": False,
        "train": [
            "List the directory {v}.", "ls {v}.", "What's in {v}?",
            "Show me the contents of {v}.", "List files in {v}.", "What files are in {v}?",
            "Show what's inside {v}.", "List everything in {v}.",
            "List out {v}.", "Show me the {v} folder.", "What's in the {v} directory?",
            "Give me a listing of {v}.", "Peek inside {v}.", "Show the {v} directory.",
        ],
        "eval": [
            "Could you list {v}?", "ls the {v} folder.", "Show the files under {v}.",
            "What's inside the {v} directory?", "Enumerate {v}.", "Display the contents of {v}.",
            "List what's in {v}.", "Print the contents of {v}.",
        ],
    },
    "find_files": {
        "group": "code", "arg": "pattern", "pool": GLOB_PATTERNS, "quoted": True,
        "train": [
            "Find files matching {v}.", "Find all {v} files.", "Locate {v} files.",
            "Search for files matching {v}.", "Which files match {v}?",
            "Glob for {v}.", "List files matching {v}.", "Find every {v} file.",
            "Find files named {v}.", "Show me files matching {v}.",
            "Dig up files matching {v}.", "Find the {v} files.", "Locate everything matching {v}.",
            "Round up files matching {v}.",
        ],
        "eval": [
            "Look for files matching {v}.", "Where are the {v} files?",
            "Find anything matching {v}.", "Get me the {v} files.",
            "Search the tree for {v}.", "Match files against {v}.",
            "Scan for files matching {v}.", "Pull up files matching {v}.",
        ],
    },
    "grep_search": {
        "group": "code", "arg": "pattern", "pool": GREP_PATTERNS, "quoted": True,
        "train": [
            "Search the codebase for {v}.", "Grep for {v}.", "Find {v} in the code.",
            "Where is {v} defined?", "Where is {v} used?", "Find the function {v}.",
            "Look for {v} in the repo.", "Locate {v} in the source.",
            "Search the code for {v}.", "Which files contain {v}?",
            "Grep the code for {v}.", "Find all references to {v}.",
            "Search the repo for {v}.", "Find occurrences of {v}.",
            "Look up {v} in the source code.", "Scan the codebase for {v}.",
        ],
        "eval": [
            "Grep the repo for {v}.", "Find where {v} appears.",
            "Hunt for {v} in the codebase.", "Search source files for {v}.",
            "Track down {v} in the code.", "Show me where {v} lives.",
            "Comb the code for {v}.", "Find every mention of {v}.",
        ],
    },
    "make_dir": {
        "group": "code", "arg": "path", "pool": DIR_PATHS, "quoted": False,
        "train": [
            "Make a directory called {v}.", "Create a folder named {v}.", "mkdir {v}.",
            "Create the directory {v}.", "Make a new folder {v}.", "Set up a directory {v}.",
            "Create folder {v}.", "Add a directory named {v}.",
            "Make the folder {v}.", "Create a directory named {v}.",
            "Set up a folder called {v}.", "New directory: {v}.",
            "Make me a directory {v}.", "Build a folder named {v}.",
        ],
        "eval": [
            "Make a folder called {v}.", "Create a new directory {v}.",
            "Spin up a folder named {v}.", "I need a directory called {v}.",
            "Please make a directory {v}.", "Add a new folder {v}.",
            "Establish a directory named {v}.", "Throw together a folder {v}.",
        ],
    },
    "run_command": {
        "group": "code", "arg": "command", "pool": SHELL_COMMANDS, "quoted": True,
        "train": [
            "Run the command {v}.", "Execute {v}.", "Run {v} in the shell.",
            "Run {v}.", "Execute the command {v} in the terminal.",
            "Shell out {v}.", "Kick off {v}.", "Fire off {v}.",
            "Run the shell command {v}.", "Type {v} in the terminal.",
            "Run {v} in the terminal.", "Execute {v} in the shell.",
            "Run this command: {v}.", "Drop {v} into the shell.",
        ],
        "eval": [
            "Could you run {v}?", "Please execute {v}.", "Run {v} for me.",
            "Invoke {v}.", "Launch {v} in the shell.", "Go ahead and run {v}.",
            "Spin up {v} in the shell.", "Run the terminal command {v}.",
        ],
    },
    "run_python": {
        "group": "code", "arg": "code", "pool": PY_SNIPPETS, "quoted": True,
        "train": [
            "Run the Python code {v}.", "Execute {v} in Python.", "Evaluate {v} in a Python shell.",
            "Run {v} in Python.", "What does {v} return in Python?",
            "Compute {v} with Python.", "Run this Python: {v}.",
            "Run the snippet {v} in Python.", "Use Python to run {v}.",
            "Fire up Python and run {v}.", "Have Python evaluate {v}.",
            "Let Python compute {v}.", "Interpret {v} as Python.",
        ],
        "eval": [
            "Execute the Python snippet {v}.", "Eval {v} in Python.",
            "Run {v} through the Python interpreter.", "Evaluate the Python {v}.",
            "Try {v} in a Python REPL.", "Give me the result of {v} in Python.",
            "Pop {v} into Python.", "Run {v} with the Python interpreter.",
        ],
    },
    "read_file": {
        "group": "code", "arg": "path", "pool": FILE_PATHS, "quoted": False,
        "train": [
            "Read the file {v}.", "Open {v}.", "Show me {v}.", "Display {v}.",
            "Cat {v}.", "Show the contents of {v}.", "Print out {v}.", "Let me see {v}.",
            "Read {v} for me.", "Bring up {v}.", "Give me the contents of {v}.",
            "Read out {v}.", "Take a look at {v} and show it to me.",
        ],
        "eval": [
            "Could you read {v}?", "Pull up the file {v}.", "Open up {v}.",
            "What's in the file {v}?", "Show me the contents of {v}.", "Dump {v}.",
            "Read me {v}.", "Print the file {v}.",
        ],
    },
    "write_file": {
        "group": "code", "arg": "path", "pool": FILE_PATHS, "quoted": False,
        "train": [
            "Create the file {v}.", "Write to {v}.", "Save the file {v}.",
            "Add a new file {v}.", "Make a file called {v}.", "Create {v}.",
            "Write out {v}.", "Generate the file {v}.",
            "Start a new file {v}.", "Write the file {v}.", "Spin up a file {v}.",
            "Author the file {v}.", "Create a brand new file {v}.",
        ],
        "eval": [
            "Could you create {v}?", "Write a new file {v}.", "Save out {v}.",
            "Make a new file named {v}.", "Create a file at {v}.", "Put together the file {v}.",
            "Draft the file {v}.", "Set up a file called {v}.",
        ],
    },
    "edit_file": {
        "group": "code", "arg": "path", "pool": FILE_PATHS, "quoted": False,
        "train": [
            "Edit {v}.", "Modify {v}.", "Make changes to {v}.", "Update the file {v}.",
            "Open {v} for editing.", "Tweak {v}.", "Change {v}.", "Revise {v}.",
            "Edit the file {v}.", "Make a quick edit to {v}.", "Touch up {v}.",
            "Fix up {v}.", "Go edit {v}.",
        ],
        "eval": [
            "Could you edit {v}?", "Adjust the file {v}.", "Patch up {v}.",
            "Make edits to {v}.", "Update {v} please.", "Alter the file {v}.",
            "Rework {v}.", "Edit up the file {v}.",
        ],
    },
    "send_email": {
        "group": "productivity", "arg": "recipient", "pool": EMAIL_NAMES, "quoted": False,
        "train": [
            "Send an email to {v}.", "Email {v}.", "Write an email to {v}.",
            "Compose an email to {v}.", "Shoot {v} an email.", "Drop {v} an email.",
            "Send {v} a message by email.", "Fire off an email to {v}.",
            "Send an email over to {v}.", "Dash off an email to {v}.",
            "Email a note to {v}.", "Send {v} an email message.", "Reach {v} by email.",
        ],
        "eval": [
            "Could you email {v}?", "Send {v} an email.", "Get an email out to {v}.",
            "Email {v} for me.", "Write to {v} by email.", "Ping {v} over email.",
            "Put together an email to {v}.", "Mail {v}.",
        ],
    },
    "calendar_event": {
        "group": "productivity", "arg": "title", "pool": CAL_TITLES, "quoted": True,
        "train": [
            "Add a calendar event called {v}.", "Schedule {v} on my calendar.",
            "Create a calendar event {v}.", "Put {v} on the calendar.",
            "Block off time for {v}.", "Add {v} to my calendar.",
            "Set up a calendar event for {v}.", "Make a calendar entry for {v}.",
            "Schedule {v}.", "Get {v} on my calendar.", "Mark down {v} on the calendar.",
            "Save {v} as a calendar event.", "Slot in {v} on my calendar.",
        ],
        "eval": [
            "Could you schedule {v}?", "Pencil in {v} on my calendar.",
            "Book {v} on the calendar.", "Add an event called {v}.",
            "Put {v} in my schedule.", "Create an appointment for {v}.",
            "Set {v} on my calendar.", "Drop {v} onto the calendar.",
        ],
    },
    "slack_send": {
        "group": "productivity", "arg": "message", "pool": SLACK_MSGS, "quoted": True,
        "train": [
            "Send a Slack message saying {v}.", "Post {v} to Slack.", "Slack the team {v}.",
            "Send {v} on Slack.", "Message the channel {v}.", "Drop a Slack note: {v}.",
            "Ping Slack with {v}.", "Post a Slack update: {v}.",
            "Slack everyone {v}.", "Send {v} to the Slack channel.",
            "Message the team on Slack: {v}.", "Fire a Slack message {v}.",
            "Let the team know on Slack: {v}.",
        ],
        "eval": [
            "Could you Slack {v}?", "Send {v} over Slack.", "Tell the team on Slack {v}.",
            "Post to Slack: {v}.", "Shoot a Slack message {v}.", "Notify Slack with {v}.",
            "Drop {v} in Slack.", "Send a Slack saying {v}.",
        ],
    },
    "run_tests": {
        "group": "code", "arg": None, "pool": None, "quoted": False,
        "train": [
            "Run the tests.", "Run the test suite.", "Execute all tests.",
            "Kick off the tests.", "Run all the unit tests.", "Let's run the tests.",
            "Fire off the test suite.", "Run the unit tests.", "Start the test run.",
            "Trigger the test suite.", "Run our tests.", "Can we run the tests?",
        ],
        "eval": [
            "Could you run the tests?", "Please run the test suite.",
            "Run all tests now.", "Go ahead and run the tests.",
            "Execute the test suite.", "Time to run the tests.",
            "Launch the test suite.", "Spin up the tests.",
        ],
    },
    "git_status": {
        "group": "code", "arg": None, "pool": None, "quoted": False,
        "train": [
            "Show the git status.", "What's the git status?", "Git status.",
            "Check the repo status.", "What's changed in the repo?", "Show me the working tree status.",
            "What's the state of the repo?", "Tell me the git status.",
            "How does the repo look?", "Any uncommitted changes?",
            "Show me the repo status.", "What's the current git state?",
        ],
        "eval": [
            "Could you check git status?", "Run git status.", "What does git status say?",
            "Give me the repo status.", "Status of the git repo?", "Show me what's staged.",
            "Pull up the git status.", "Where does the repo stand?",
        ],
    },
    "git_diff": {
        "group": "code", "arg": None, "pool": None, "quoted": False,
        "train": [
            "Show the git diff.", "What's the diff?", "Git diff.",
            "Show me the changes.", "Display the current diff.", "What did I change?",
            "Let me see the diff.", "Show the changes I made.", "What's different?",
            "Print the diff.", "Show me what changed.", "Give me the diff.",
        ],
        "eval": [
            "Could you show the diff?", "Run git diff.", "What's the current diff?",
            "Show me the unstaged changes.", "Give me the git diff.", "Diff the working tree.",
            "Pull up the diff.", "What edits are pending?",
        ],
    },
    "kill_process": {
        "group": "code", "arg": "name", "pool": PROC_NAMES, "quoted": True,
        "train": [
            "Kill the process {v}.", "Kill {v}.", "Stop the {v} process.",
            "Terminate {v}.", "End the {v} process.", "Shut down {v}.",
            "Force quit {v}.", "Kill off {v}.",
            "Kill the {v} process.", "Knock out {v}.", "Bring down the {v} process.",
            "Stop {v} from running.", "Wipe out the {v} process.",
        ],
        "eval": [
            "Could you kill {v}?", "Stop {v}.", "Take down the {v} process.",
            "Halt {v}.", "Terminate the process {v}.", "Quit {v}.",
            "Put an end to {v}.", "Shut {v} down.",
        ],
    },
    "install_package": {
        "group": "code", "arg": "name", "pool": PKG_NAMES, "quoted": True,
        "train": [
            "Install {v}.", "Install the package {v}.", "Add the dependency {v}.",
            "Set up {v}.", "Pull in {v}.", "Add {v} to the project.",
            "Get {v} installed.", "Bring in the package {v}.",
            "Install the {v} package.", "Add {v} as a dependency.",
            "Download and install {v}.", "Drop in the {v} package.", "Wire up {v}.",
        ],
        "eval": [
            "Could you install {v}?", "Add the package {v}.", "Install the dependency {v}.",
            "Set up the {v} package.", "Grab {v}.", "Install {v} please.",
            "Pull down {v}.", "Get me the {v} package.",
        ],
    },
    # --- remaining tools (still paraphrase-varied, fewer synonyms needed) --------------------
    "get_weather": {
        "group": "tool_call", "arg": "city", "pool": {
            "train": ["Sydney", "Auckland", "Geneva", "Casablanca", "Marrakech", "Lyon",
                      "Hamburg", "Florence", "Galway", "Cork", "Bristol", "Cardiff"],
            "eval": ["Wellington", "Zurich", "Rabat", "Nice", "Cologne", "Verona"],
        }, "quoted": False,
        "train": ["What's the weather in {v}?", "How's the weather in {v}?",
                  "Is it raining in {v}?", "What's it like outside in {v}?",
                  "Tell me the weather for {v}.", "Forecast for {v}?",
                  "How hot is it in {v}?", "Do I need an umbrella in {v}?",
                  "What's the weather like in {v} today?", "Check the weather in {v}.",
                  "Is it warm in {v}?", "Pull up the weather for {v}."],
        "eval": ["What's the forecast in {v}?", "Weather in {v}?",
                 "Is it sunny in {v}?", "How cold is it in {v}?",
                 "Give me the weather for {v}.", "What's the temperature in {v}?",
                 "How's it looking outside in {v}?", "Is it windy in {v}?"],
    },
    "calculator": {
        "group": "tool_call", "arg": "expression", "pool": {
            "train": ["12+7", "9*6", "20-4", "15+8", "7*7", "100-33", "6*9", "44+11"],
            "eval": ["18+5", "8*8", "50-17", "13*3", "9+24"],
        }, "quoted": False,
        "train": ["What is {v}?", "How much is {v}?", "Compute {v}.",
                  "Calculate {v}.", "What's {v}?", "Work out {v}.", "Add up {v}.",
                  "Tell me what {v} is.", "Do the math on {v}.", "Give me {v}.",
                  "Quick, what's {v}?", "Sum up {v}."],
        "eval": ["Can you compute {v}?", "Figure out {v}.", "Solve {v}.",
                 "Evaluate {v}.", "What does {v} equal?", "Crunch {v}.",
                 "Run the numbers on {v}.", "What's the answer to {v}?"],
    },
    "planner": {
        "group": "planner", "arg": "goal", "pool": {
            "train": ["launch a newsletter", "learn to code", "grow a garden",
                      "train for a triathlon", "write a novel", "build a treehouse",
                      "start a business", "learn Spanish", "save for a house",
                      "organize the office", "plan a road trip", "get into shape"],
            "eval": ["learn photography", "open a cafe", "build an app",
                     "run a half marathon", "plant an orchard", "start a blog"],
        }, "quoted": False,
        "train": ["Make a plan to {v}.", "I want to {v}.", "I need to {v}.",
                  "Help me {v}.", "Plan how to {v}.", "Create a plan to {v}.",
                  "My goal is to {v}.", "Put together a plan to {v}.",
                  "Figure out how to {v}.", "Help me plan to {v}.",
                  "Draw up a plan to {v}.", "Give me a plan to {v}."],
        "eval": ["Could you plan how to {v}?", "Lay out a plan to {v}.",
                 "Walk me through how to {v}.", "Map out how to {v}.",
                 "I'd like to {v}.", "Outline steps to {v}.",
                 "Sketch a plan to {v}.", "Plan out how to {v}."],
    },
    "play_music": {
        "group": "music", "arg": "song", "pool": {
            "train": ["Levitating", "Blinding Lights", "Shape of You", "Bad Guy",
                      "Uptown Funk", "Happy", "Rolling in the Deep", "Counting Stars",
                      "Radioactive", "Pompeii", "Royals", "Despacito"],
            "eval": ["Shake It Off", "Someone Like You", "Believer", "Demons",
                     "Stay", "Sunflower"],
        }, "quoted": False,
        "train": ["Play {v}.", "Put on {v}.", "Start playing {v}.", "Queue up {v}.",
                  "I want to hear {v}.", "Play the song {v}.", "Let's listen to {v}.",
                  "Crank up {v}.", "Can you play {v}?", "Stream {v}.",
                  "Pop on {v}.", "Play some {v}."],
        "eval": ["Could you play {v}?", "Throw on {v}.", "Spin up {v}.",
                 "Play me {v}.", "I'm in the mood for {v}.", "Hit play on {v}.",
                 "Blast {v}.", "Get {v} going."],
    },
    "set_reminder": {
        "group": "tool_call", "arg": "task", "pool": {
            "train": ["take out the trash", "feed the cat", "stretch", "drink water",
                      "stand up", "check email", "lock the door", "take vitamins",
                      "call mom", "send the invoice", "back up files", "log hours"],
            "eval": ["water the garden", "charge the phone", "submit timesheet",
                     "walk the dog", "pay the bill", "stretch my legs"],
        }, "quoted": False,
        "train": ["Set a reminder to {v}.", "Remind me to {v}.", "Add a reminder to {v}.",
                  "Don't let me forget to {v}.", "I need to remember to {v}.",
                  "Nudge me to {v}.", "Remind me later to {v}.",
                  "Set up a reminder to {v}.", "Help me remember to {v}.",
                  "Jog my memory to {v}.", "Create a reminder to {v}.",
                  "Make a reminder to {v}."],
        "eval": ["Could you remind me to {v}?", "Set a reminder for me to {v}.",
                 "Make sure I {v}.", "Ping me to {v}.", "Remind me about {v}.",
                 "Put a reminder to {v}.", "Drop a reminder to {v}.",
                 "Don't let me forget about {v}."],
    },
    "set_timer": {
        "group": "tool_call", "arg": "duration", "pool": {
            "train": ["7 minutes", "50 seconds", "35 minutes", "5 hours", "18 minutes",
                      "100 seconds", "55 minutes", "9 hours"],
            "eval": ["3 minutes", "65 seconds", "7 hours", "22 minutes", "110 seconds"],
        }, "quoted": False,
        "train": ["Set a timer for {v}.", "Start a timer for {v}.", "Set a countdown for {v}.",
                  "Wake me in {v}.", "Let me know in {v}.", "Ping me in {v}.",
                  "Time {v} for me.", "Put a timer on for {v}.", "Kick off a timer for {v}.",
                  "Remind me in {v}.", "Set the timer to {v}.", "Start counting down {v}."],
        "eval": ["Could you set a timer for {v}?", "Count down {v}.",
                 "Buzz me in {v}.", "Run a timer for {v}.", "Alert me in {v}.",
                 "Give me {v} on the timer.", "Notify me in {v}.", "Tick down {v}."],
    },
    "notion_write": {
        "group": "productivity", "arg": "content", "pool": {
            "train": ["sprint retro notes", "groceries list", "vacation plans",
                      "book recommendations", "workout log", "expense tracker",
                      "movie watchlist", "recipe ideas", "habit tracker", "gift ideas",
                      "travel checklist", "study schedule"],
            "eval": ["meeting recap", "project roadmap", "packing list",
                     "reading goals", "budget notes", "todo for today"],
        }, "quoted": True,
        "train": ["Write {v} in Notion.", "Add a Notion note saying {v}.",
                  "Note {v} in Notion.", "Save {v} to Notion.", "Jot {v} into Notion.",
                  "Log {v} in Notion.", "Put {v} in my Notion.",
                  "Add {v} as a Notion note.", "Write up {v} in Notion.",
                  "Stick {v} into Notion."],
        "eval": ["Could you note {v} in Notion?", "Drop {v} into Notion.",
                 "Add {v} to Notion.", "Record {v} in Notion.",
                 "Save a Notion note: {v}.", "Write down {v} in Notion.",
                 "Pop {v} into Notion.", "Capture {v} in Notion."],
    },
    "jira_issue": {
        "group": "productivity", "arg": "summary", "pool": {
            "train": ["checkout fails", "slow page load", "missing translations",
                      "broken search", "wrong totals", "image upload error",
                      "duplicate emails", "timezone bug", "stale cache", "crash on logout",
                      "bad redirect", "missing icons"],
            "eval": ["payment timeout", "blank dashboard", "export broken",
                     "wrong currency", "stuck spinner", "lost session"],
        }, "quoted": True,
        "train": ["Create a Jira ticket titled {v}.", "Open a Jira issue for {v}.",
                  "File a Jira bug for {v}.", "Log a Jira issue: {v}.",
                  "Raise a Jira ticket for {v}.", "Make a Jira issue about {v}.",
                  "Track {v} in Jira.", "Spin up a Jira ticket for {v}.",
                  "Report {v} in Jira.", "Add a Jira ticket titled {v}."],
        "eval": ["Could you open a Jira issue for {v}?", "File a ticket for {v}.",
                 "Create a bug report for {v}.", "Log {v} as a Jira ticket.",
                 "Open a Jira bug: {v}.", "Add a Jira issue for {v}.",
                 "Cut a Jira ticket for {v}.", "Submit a Jira issue about {v}."],
    },
    "git_commit": {
        "group": "code", "arg": "message", "pool": {
            "train": ["initial commit", "wip", "cleanup", "add feature flag",
                      "fix lint", "update readme", "bump deps", "add migration",
                      "tweak styles", "refactor api", "add logging", "fix build"],
            "eval": ["polish ui", "drop legacy code", "patch security", "add caching",
                     "wire up auth", "fix flaky test"],
        }, "quoted": True,
        "train": ["Commit with message {v}.", "Make a git commit saying {v}.",
                  "Git commit with {v}.", "Commit the changes: {v}.",
                  "Create a commit {v}.", "Commit as {v}.", "Save a commit {v}.",
                  "Commit everything with {v}.", "Make a commit with the message {v}.",
                  "Write a commit {v}."],
        "eval": ["Could you commit with {v}?", "Make a commit titled {v}.",
                 "Commit it as {v}.", "Record a commit {v}.",
                 "Git commit message {v}.", "Stage and commit {v}.",
                 "Commit the work with {v}.", "Land a commit {v}."],
    },
    "apply_patch": {
        "group": "code", "arg": "path", "pool": FILE_PATHS, "quoted": False,
        "train": ["Apply the patch to {v}.", "Patch {v}.", "Apply a patch to the file {v}.",
                  "Patch the file {v}.", "Apply changes to {v} as a patch.",
                  "Run the patch on {v}.", "Patch up the file {v}.",
                  "Apply the diff to {v}.", "Lay the patch over {v}.",
                  "Patch in changes to {v}."],
        "eval": ["Could you patch {v}?", "Apply a diff to {v}.",
                 "Patch the file at {v}.", "Run a patch against {v}.",
                 "Apply the patch onto {v}.", "Merge the patch into {v}.",
                 "Roll the patch into {v}.", "Stitch the patch into {v}."],
    },
    "sql_query": {
        "group": "code", "arg": "query", "pool": {
            "train": ["SELECT * FROM clients", "SELECT count(*) FROM sales",
                      "DELETE FROM sessions", "SELECT name FROM teams",
                      "UPDATE users SET role = 'admin'", "SELECT * FROM invoices",
                      "INSERT INTO logs VALUES (1)", "SELECT id FROM tickets"],
            "eval": ["SELECT * FROM vendors", "SELECT sum(total) FROM orders",
                     "TRUNCATE temp_data", "SELECT email FROM leads",
                     "UPDATE jobs SET done = 1", "SELECT * FROM audits"],
        }, "quoted": True,
        "train": ["Run the SQL query {v}.", "Execute {v} on the database.",
                  "Query the database with {v}.", "Run {v} against the db.",
                  "Fire {v} at the database.", "Run this SQL: {v}.", "Execute the query {v}.",
                  "Send {v} to the database.", "Run the query {v} on the db.",
                  "Execute the SQL statement {v}."],
        "eval": ["Could you run the SQL {v}?", "Run {v} on the db.",
                 "Execute {v} against the database.", "Query with {v}.",
                 "Run the query {v}.", "Hit the database with {v}.",
                 "Throw {v} at the database.", "Run SQL {v} now."],
    },
    "env_get": {
        "group": "code", "arg": "name", "pool": {
            "train": ["NODE_ENV", "API_URL", "PORT", "DB_HOST", "CACHE_TTL", "LOG_DIR",
                      "MAX_RETRIES", "QUEUE_NAME", "REGION", "BUILD_ID", "APP_NAME", "TOKEN"],
            "eval": ["DB_PORT", "SMTP_HOST", "WORKER_COUNT", "BASE_URL", "FLAG_X", "RUN_MODE"],
        }, "quoted": True,
        "train": ["Get the env variable {v}.", "Read the environment variable {v}.",
                  "What is {v} set to?", "Show the value of {v}.",
                  "Print the env var {v}.", "Look up the env variable {v}.",
                  "Fetch the value of {v}.", "Tell me what {v} is set to.",
                  "Grab the env var {v}.", "Check the environment variable {v}."],
        "eval": ["Could you read the env var {v}?", "What's the value of {v}?",
                 "Get me the environment variable {v}.", "Echo {v}.",
                 "Show the env variable {v}.", "Read {v} from the environment.",
                 "Print out the value of {v}.", "Look up {v} in the env."],
    },
    "docker_run": {
        "group": "code", "arg": "image", "pool": {
            "train": ["caddy", "vault", "consul", "minio", "loki", "grafana:latest",
                      "node:18", "python:3.11", "ruby:3", "golang:1.22", "openjdk", "haproxy"],
            "eval": ["nats", "etcd", "tempo", "jaeger", "kong", "vault:latest"],
        }, "quoted": True,
        "train": ["Run a Docker container from {v}.", "Run the {v} image.",
                  "Start a container from {v}.", "docker run {v}.",
                  "Spin up a container from {v}.", "Launch the {v} container.",
                  "Boot up {v} in Docker.", "Start the {v} image in Docker.",
                  "Run {v} as a container.", "Stand up a container from {v}."],
        "eval": ["Could you run the {v} image?", "Start the container {v}.",
                 "Bring up a container from {v}.", "Run {v} in Docker.",
                 "Fire up the {v} image.", "Deploy a container from {v}.",
                 "Kick off the {v} container.", "Pull and run {v}."],
    },
    "unzip": {
        "group": "code", "arg": "path", "pool": {
            "train": ["release.zip", "data.tar.gz", "assets.zip", "logs.zip", "models.zip",
                      "src.zip", "images.tar.gz", "bundle.zip", "exports.zip", "docs.zip",
                      "fonts.zip", "samples.zip"],
            "eval": ["vendor.zip", "photos.tar.gz", "weights.zip", "backup.zip",
                     "fixtures.zip", "templates.zip"],
        }, "quoted": False,
        "train": ["Unzip {v}.", "Extract {v}.", "Unpack the archive {v}.",
                  "Decompress {v}.", "Open up the archive {v}.", "Expand {v}.",
                  "Unpack {v}.", "Unzip the archive {v}.", "Extract the files from {v}.",
                  "Unpack the file {v}."],
        "eval": ["Could you unzip {v}?", "Pull apart {v}.", "Extract the archive {v}.",
                 "Uncompress {v}.", "Unbundle {v}.", "Open the zip {v}.",
                 "Crack open {v}.", "Extract everything from {v}."],
    },
    "write_clipboard": {
        "group": "computer_use", "arg": "text", "pool": {
            "train": ["see you soon", "order 7781", "draft message", "phone number",
                      "tracking code", "promo code", "wifi password", "address line",
                      "session id", "ticket number", "license key", "coupon code"],
            "eval": ["meeting link", "confirmation code", "api key", "reference number",
                     "access token", "booking id"],
        }, "quoted": True,
        "train": ["Copy {v} to the clipboard.", "Put {v} on the clipboard.", "Copy {v}.",
                  "Set the clipboard to {v}.", "Save {v} to the clipboard.",
                  "Stick {v} on the clipboard.", "Copy the text {v}.",
                  "Throw {v} on the clipboard.", "Add {v} to the clipboard.",
                  "Copy {v} for me."],
        "eval": ["Could you copy {v}?", "Place {v} on the clipboard.",
                 "Copy over {v}.", "Load {v} into the clipboard.",
                 "Put the text {v} on the clipboard.", "Clipboard {v}.",
                 "Drop {v} onto the clipboard.", "Store {v} in the clipboard."],
    },
    "type_text": {
        "group": "computer_use", "arg": "text", "pool": {
            "train": ["hello there", "user@site.com", "search term", "full name",
                      "the lazy dog", "March 2026", "secret pass", "home address",
                      "monthly report", "daily notes", "order 3310", "billing info"],
            "eval": ["good evening", "info@mail.com", "yearly plan", "patch notes",
                     "agenda items", "client notes"],
        }, "quoted": True,
        "train": ["Type {v}.", "Type {v} into the field.", "Enter {v}.",
                  "Input {v}.", "Write {v} in the box.", "Fill in {v}.", "Key in {v}.",
                  "Type in {v}.", "Type the text {v}.", "Enter {v} into the field."],
        "eval": ["Could you type {v}?", "Put {v} in the field.", "Punch in {v}.",
                 "Enter the text {v}.", "Fill the box with {v}.", "Type out {v}.",
                 "Tap in {v}.", "Write out {v} in the field."],
    },
    "click": {
        "group": "computer_use", "arg": "target", "pool": {
            "train": ["the Save button", "the Login link", "the menu icon", "the search bar",
                      "the OK button", "the plus button", "the avatar", "the tab",
                      "the toggle", "the arrow", "the logo", "the banner"],
            "eval": ["the trash icon", "the star button", "the back link", "the dropdown",
                     "the checkbox", "the close icon"],
        }, "quoted": True,
        "train": ["Click {v}.", "Click on {v}.", "Press {v}.", "Tap {v}.", "Hit {v}.",
                  "Select {v}.", "Go ahead and click {v}.",
                  "Click {v} please.", "Mouse over and click {v}.", "Just click {v}."],
        "eval": ["Could you click {v}?", "Give {v} a click.", "Push {v}.",
                 "Click the {v}.", "Tap on {v}.", "Choose {v}.",
                 "Press on {v}.", "Left-click {v}."],
    },
    "double_click": {
        "group": "computer_use", "arg": "target", "pool": {
            "train": ["the folder", "the file", "the app icon", "the word", "the cell",
                      "the row", "the thumbnail", "the shortcut", "the layer", "the node",
                      "the slide", "the track"],
            "eval": ["the document", "the picture", "the entry", "the item",
                     "the desktop icon", "the tile"],
        }, "quoted": True,
        "train": ["Double-click {v}.", "Double click {v}.", "Double-click on {v}.",
                  "Open {v} by double-clicking.", "Double tap {v}.",
                  "Twice-click {v}.", "Double-click the {v}.",
                  "Double-click {v} to open it.", "Hit {v} twice.", "Two clicks on {v}."],
        "eval": ["Could you double-click {v}?", "Give {v} a double click.",
                 "Double-tap on {v}.", "Open {v} with a double click.",
                 "Double click on the {v}.", "Quickly double-click {v}.",
                 "Tap {v} twice.", "Do a double-click on {v}."],
    },
    "key_press": {
        "group": "computer_use", "arg": "key", "pool": {
            "train": ["Enter", "Tab", "Escape", "Space"],
            "eval": ["Backspace", "Delete", "ArrowUp", "ArrowDown"],
        }, "quoted": False,
        "train": ["Press {v}.", "Hit the {v} key.", "Press the {v} key.",
                  "Tap {v}.", "Send a {v} keypress.", "Stroke {v}.", "Key {v}.",
                  "Hit {v}.", "Type the {v} key.", "Press {v} now."],
        "eval": ["Could you press {v}?", "Smash {v}.", "Push the {v} key.",
                 "Trigger {v}.", "Press down {v}.", "Tap the {v} key.",
                 "Strike {v}.", "Send the {v} key."],
    },
    "scroll": {
        "group": "computer_use", "arg": "direction", "pool": {
            "train": ["up", "down"],
            "eval": ["left", "right"],
        }, "quoted": False,
        "train": ["Scroll {v}.", "Scroll {v} a bit.", "Scroll the page {v}.",
                  "Please scroll {v}.", "Keep scrolling {v}.", "Move {v} on the page.",
                  "Roll {v}.", "Scroll {v} some more.", "Drag the page {v}.", "Swipe {v}."],
        "eval": ["Could you scroll {v}?", "Scroll a little {v}.", "Nudge the page {v}.",
                 "Scroll further {v}.", "Pan {v}.", "Go {v} on the page.",
                 "Scroll {v} on the screen.", "Push the page {v}."],
    },
    "drag": {
        "group": "computer_use", "arg": None, "pool": None, "quoted": True,
        # drag needs two targets; handled specially in _make_drag below
        "train": [], "eval": [],
    },
    "wait": {
        "group": "computer_use", "arg": "seconds", "pool": {
            "train": [6, 11, 14, 22, 35, 50, 80, 100],
            "eval": [9, 13, 28, 42, 70, 95],
        }, "quoted": False, "int": True,
        "train": ["Wait {v} seconds.", "Wait for {v} seconds.", "Pause for {v} seconds.",
                  "Hold on {v} seconds.", "Give it {v} seconds.", "Sleep {v} seconds.",
                  "Hang on {v} seconds.", "Wait around {v} seconds.",
                  "Hold for {v} seconds.", "Sit tight {v} seconds."],
        "eval": ["Could you wait {v} seconds?", "Pause {v} seconds.", "Wait {v} secs.",
                 "Stall for {v} seconds.", "Linger {v} seconds.", "Delay {v} seconds.",
                 "Idle for {v} seconds.", "Wait out {v} seconds."],
    },
    "move_cursor": {
        "group": "computer_use", "arg": "target", "pool": {
            "train": ["the Save button", "the Login link", "the menu icon", "the search bar",
                      "the OK button", "the plus button", "the avatar", "the tab",
                      "the toggle", "the arrow", "the logo", "the banner"],
            "eval": ["the trash icon", "the star button", "the back link", "the dropdown",
                     "the checkbox", "the close icon"],
        }, "quoted": True,
        "train": ["Move the cursor to {v}.", "Hover over {v}.", "Move the mouse to {v}.",
                  "Point at {v}.", "Bring the cursor to {v}.", "Float over {v}.",
                  "Position the cursor on {v}.", "Slide the cursor to {v}.",
                  "Put the cursor on {v}.", "Drift the mouse over to {v}."],
        "eval": ["Could you hover over {v}?", "Move the pointer to {v}.",
                 "Glide the cursor to {v}.", "Rest the mouse on {v}.",
                 "Hover the cursor over {v}.", "Send the cursor to {v}.",
                 "Drag the pointer to {v}.", "Aim the cursor at {v}."],
    },
    "open_app": {
        "group": "computer_use", "arg": "name", "pool": {
            "train": ["Brave", "Edge", "Sublime", "iMovie", "Keynote", "Pages",
                      "Numbers", "Reminders", "Maps", "Music", "Books", "Stocks"],
            "eval": ["Vivaldi", "Atom", "GarageBand", "Photos", "Clock", "Weather"],
        }, "quoted": True,
        "train": ["Open {v}.", "Launch {v}.", "Open the {v} app.", "Start {v}.",
                  "Fire up {v}.", "Bring up {v}.", "Open up {v}.",
                  "Launch the {v} app.", "Start up {v}.", "Kick off the {v} app."],
        "eval": ["Could you open {v}?", "Boot up {v}.", "Start the {v} app.",
                 "Pull up {v}.", "Get {v} open.", "Run the app {v}.",
                 "Open the app {v}.", "Fire up the {v} app."],
    },
    "screenshot": {
        "group": "computer_use", "arg": None, "pool": None, "quoted": False,
        "train": ["Take a screenshot.", "Capture the screen.",
                  "Grab a screenshot of the screen.", "Screenshot the current screen.",
                  "Snap a picture of what's on screen.", "Grab the screen.",
                  "Take a screen capture.", "Capture a screenshot.",
                  "Take a shot of the screen.", "Save a screenshot."],
        "eval": ["Could you take a screenshot?", "Snap the screen.",
                 "Capture what's on screen.", "Get a screenshot.",
                 "Screenshot this.", "Take a picture of the screen.",
                 "Grab a screen capture.", "Snap a screenshot for me."],
    },
    "read_clipboard": {
        "group": "computer_use", "arg": None, "pool": None, "quoted": False,
        "train": ["Read the clipboard.", "What's on the clipboard?",
                  "Get the clipboard contents.", "Paste the clipboard.",
                  "Show me what's copied.", "Read what's in the clipboard.",
                  "Check the clipboard.", "What did I copy?",
                  "Print the clipboard.", "Read out the clipboard."],
        "eval": ["Could you read the clipboard?", "What's copied?",
                 "Grab the clipboard contents.", "Show the clipboard.",
                 "Tell me what's on the clipboard.", "Fetch the clipboard.",
                 "What's in the clipboard right now?", "Pull the clipboard contents."],
    },
    "list_processes": {
        "group": "code", "arg": None, "pool": None, "quoted": False,
        "train": ["List the running processes.", "Show running processes.",
                  "What processes are running?", "List all processes.",
                  "Show me the process list.", "List the processes.",
                  "What's running right now?", "Print the running processes.",
                  "Show all running processes.", "List out the processes."],
        "eval": ["Could you list the processes?", "Show me running processes.",
                 "Which processes are active?", "Display all processes.",
                 "Give me the process list.", "What processes are up?",
                 "Enumerate the running processes.", "List the active processes."],
    },
}


def _all_tool_names() -> list[str]:
    return [t.name for t in STANDARD_TOOLS]


def _format(template: str, value, quoted: bool) -> str:
    if quoted:
        return template.replace("{v}", f"'{value}'")
    return template.replace("{v}", str(value))


def _make_drag(rng: random.Random, split: str) -> Sample:
    targets = (["the Save button", "the Login link", "the menu icon", "the search bar",
                "the plus button", "the avatar"] if split == "train"
               else ["the trash icon", "the star button", "the dropdown", "the checkbox"])
    src, dst = rng.sample(targets, 2)
    tmpls = (["Drag {s} to {d}.", "Drag {s} onto {d}.",
              "Move {s} over to {d} by dragging.", "Drag and drop {s} to {d}.",
              "Drag {s} over onto {d}.", "Grab {s} and drag it to {d}.",
              "Drag {s} all the way to {d}.", "Slide {s} onto {d}."] if split == "train"
             else ["Could you drag {s} to {d}?", "Drag {s} across to {d}.",
                   "Pull {s} onto {d}.", "Drag {s} and drop it on {d}.",
                   "Drag {s} right onto {d}.", "Haul {s} over to {d}."])
    t = rng.choice(tmpls)
    prompt = t.replace("{s}", f"'{src}'").replace("{d}", f"'{dst}'")
    args = {"source": src, "dest": dst}
    return Sample("drag", "computer_use", prompt, "tool",
                  _tool_target("drag", args), "drag",
                  json.dumps(args, separators=(",", ":"), sort_keys=True))


def _make_one(name: str, spec: dict, rng: random.Random, split: str) -> Sample:
    if name == "drag":
        return _make_drag(rng, split)
    templates = spec[split]
    quoted = spec.get("quoted", False)
    group = spec["group"]
    arg = spec["arg"]
    if arg is None:                                   # arg-less tool
        prompt = rng.choice(templates)
        args: dict = {}
    else:
        pool = _pool(spec["pool"], split)
        value = rng.choice(pool)
        prompt = _format(rng.choice(templates), value, quoted)
        args = {arg: int(value) if spec.get("int") else value}
    return Sample(name, group, prompt, "tool",
                  _tool_target(name, args), name,
                  json.dumps(args, separators=(",", ":"), sort_keys=True))


def paraphrase_samples(n: int = 1, seed: int = 0, split: str = "train") -> list[Sample]:
    """Return paraphrase-rich tool Samples, balanced across all 50 ``STANDARD_TOOLS``.

    ``n`` samples are produced per tool (so ``len`` == ``n * 50``). ``split="eval"`` draws from the
    DISJOINT held phrasing templates and slot pools, measuring intent->tool generalization.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    rng = random.Random(seed)
    out: list[Sample] = []
    for name in _all_tool_names():
        spec = TOOL_SPECS[name]
        for _ in range(n):
            out.append(_make_one(name, spec, rng, split))
    rng.shuffle(out)
    return out


# Representative TRAIN-pool phrasings per tool — seed for example-query tool embeddings. Built once
# from the first few train templates, with a sample slot value substituted (literal, groundable).
def _build_tool_examples() -> dict[str, list[str]]:
    rng = random.Random(12345)
    ex: dict[str, list[str]] = {}
    for name in _all_tool_names():
        spec = TOOL_SPECS[name]
        if name == "drag":
            ex[name] = ["Drag 'the Save button' to 'the search bar'.",
                        "Drag and drop 'the avatar' to 'the plus button'."]
            continue
        templates = spec["train"][:5]
        quoted = spec.get("quoted", False)
        arg = spec["arg"]
        examples = []
        if arg is None:
            examples = list(templates)
        else:
            pool = _pool(spec["pool"], "train")
            for t in templates:
                examples.append(_format(t, rng.choice(pool), quoted))
        ex[name] = examples
    return ex


TOOL_EXAMPLES: dict[str, list[str]] = _build_tool_examples()
