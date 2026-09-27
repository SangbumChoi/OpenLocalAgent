"""Synthetic agent-data generation (Phase 3, implemented — self-contained, no teacher API).

Produces deterministic, canonical tool-calling + text samples across categories:
  weather / calc  -> tool-calling group
  web_search      -> web search
  planner         -> planner
  text / no_tool  -> text generation (incl. correct abstention, Hammer-style)

Determinism matters: each input maps to ONE canonical target string (compact, sorted-key JSON
for tool calls), so a tiny byte-level model can actually reproduce it exactly — that's what makes
~100% reachable and exactly-scorable.

Generalization, not memorization: named train/eval slot pools (cities, names, numbers, …) are
disjoint. Schema enums, no-argument intents, and some template vocabulary are deliberately shared,
so the generator makes no blanket primitive-value or template-disjointness claim. Configured
external benchmark prompts are held out by exact normalized user-prompt match.

The flywheel enriches by raising ``level`` (1..5): more templates, bigger slot pools, and a bit
more structural complexity (weather units, multi-term arithmetic, harder phrasings).
"""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass
from typing import ClassVar

from localagent.data.agent_synth_paper_v2 import (
    PAPER_TRAIN_V2_MODE,
    PAPER_TRAIN_V2_MODE_VERSION,
    PAPER_TRAIN_V2_SLOT_POOLS,
    PAPER_V2_SCROLL_AMOUNTS_TRAIN,
    PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN,
    build_paper_train_v2_tools,
)
from localagent.data.schema import Conversation, Message, Role, ToolCall


def _canonical_holdout_prompt(value: str) -> str:
    """Return the canonical, case-insensitive form used by every prompt-holdout check.

    NFKC merges compatibility-equivalent Unicode forms, ``split``/``join`` collapses all Unicode
    whitespace to one ASCII space, and Unicode ``casefold`` prevents casing-only leakage.
    """

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


# ---- slot pools, split train/eval (disjoint) ------------------------------------------------
CITIES_TRAIN = [
    "Paris",
    "Tokyo",
    "Berlin",
    "Cairo",
    "Lima",
    "Oslo",
    "Delhi",
    "Madrid",
    "Seoul",
    "Rome",
    "Dublin",
    "Vienna",
    "Athens",
    "Bogota",
    "Hanoi",
    "Accra",
    "Lisbon",
    "Prague",
    "Warsaw",
    "Helsinki",
    "Manila",
    "Jakarta",
    "Amman",
    "Doha",
    "Brussels",
    "Stockholm",
    "Budapest",
    "Belgrade",
    "Tbilisi",
    "Baku",
    "Tehran",
    "Karachi",
    "Lagos",
    "Nairobi",
    "Bangkok",
    "Taipei",
    "Osaka",
    "Munich",
    "Naples",
    "Porto",
    "Valencia",
    "Glasgow",
    "Leeds",
    "Ottawa",
    "Calgary",
    "Denver",
    "Austin",
    "Seattle",
    "Portland",
    "Phoenix",
    "Dallas",
    "Atlanta",
    "Miami",
    "Houston",
    "Toronto",
    "Montreal",
    "Santiago",
    "Brasilia",
    "Quebec",
    "Medellin",
    "Panama",
]
CITIES_EVAL = [
    "Boston",
    "Quito",
    "Kyoto",
    "Bern",
    "Tunis",
    "Riga",
    "Perth",
    "Mombasa",
    "Caracas",
    "Bruges",
    "Cusco",
    "Almaty",
    "Reykjavik",
    "Tallinn",
    "Vilnius",
    "Sarajevo",
    "Tirana",
    "Maputo",
    "Kigali",
    "Dakar",
]
NAMES_TRAIN = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Mallory",
    "Niaj",
    "Olivia",
    "Peggy",
    "Sybil",
    "Trent",
    "Karl",
    "Nora",
    "Pablo",
    "Ruth",
    "Vince",
    "Wendy",
    "Hugo",
    "Iris",
    "Aaron",
    "Bianca",
    "Cedric",
    "Dora",
    "Elliot",
    "Fiona",
    "George",
    "Hannah",
    "Isaac",
    "Jasmine",
    "Kevin",
    "Laura",
    "Marcus",
    "Nadia",
    "Oscar",
    "Priya",
    "Quentin",
    "Rachel",
    "Samuel",
    "Tara",
    "Ulises",
    "Vera",
    "Wesley",
    "Yara",
]
NAMES_EVAL = [
    "Walter",
    "Xena",
    "Yuki",
    "Zara",
    "Quinn",
    "Rosa",
    "Feliks",
    "Mira",
    "Otto",
    "Greta",
    "Diego",
    "Lena",
    "Soren",
    "Ingrid",
    "Mateo",
    "Noor",
]
QUERIES_TRAIN = [
    "best pizza in town",
    "history of jazz",
    "how tall is Everest",
    "python list comprehension",
    "tides tomorrow",
    "nearest pharmacy",
    "speed of light",
    "who wrote Hamlet",
    "capital of Peru",
    "boiling point of water",
    "longest river in Asia",
    "how to tie a tie",
    "weather on Mars",
    "population of Canada",
    "how do magnets work",
    "best running shoes",
    "cheapest flights to Rome",
    "symptoms of the flu",
    "how to make sourdough",
    "current bitcoin price",
    "rules of cricket",
    "tallest building in the world",
    "how planes fly",
    "history of the internet",
    "best sci-fi movies",
]
QUERIES_EVAL = [
    "origin of tea",
    "rules of chess",
    "how rainbows form",
    "fastest land animal",
    "phases of the moon",
    "history of the violin",
    "how vaccines work",
    "deepest part of the ocean",
    "who invented radio",
    "best hiking trails",
]
GOALS_TRAIN = [
    "plan a trip to the coast",
    "organize a birthday party",
    "learn to bake bread",
    "set up a home garden",
    "prepare for an exam",
    "build a bookshelf",
    "train for a 5k",
    "write a short story",
    "redecorate the living room",
    "save for a vacation",
    "declutter the garage",
    "start a vegetable patch",
    "learn to paint",
    "plan a wedding",
    "build a website",
    "run a marathon",
    "read more books",
    "cook healthier meals",
    "learn to swim",
    "start journaling",
]
GOALS_EVAL = [
    "plan a museum visit",
    "start a podcast",
    "fix a leaky faucet",
    "learn guitar",
    "host a dinner party",
    "switch careers",
    "adopt a puppy",
    "renovate the kitchen",
]
TERMS_TRAIN = [
    "photosynthesis",
    "inflation",
    "entropy",
    "recursion",
    "osmosis",
    "democracy",
    "gravity",
    "metaphor",
    "algorithm",
    "ecosystem",
    "capitalism",
    "evolution",
    "momentum",
    "encryption",
    "diffusion",
    "polymer",
    "tectonics",
    "antibody",
    "induction",
    "sarcasm",
    "monopoly",
    "velocity",
    "habitat",
    "irony",
]
TERMS_EVAL = [
    "mitosis",
    "diplomacy",
    "friction",
    "syntax",
    "biome",
    "catalyst",
    "satire",
    "inertia",
    "symbiosis",
    "federalism",
]
SONGS_TRAIN = [
    "Bohemian Rhapsody",
    "Hey Jude",
    "Yesterday",
    "Imagine",
    "Hallelujah",
    "Thunderstruck",
    "Clocks",
    "Africa",
    "Viva La Vida",
    "Wonderwall",
    "Stairway",
    "Billie Jean",
    "Rolling Stone",
    "Sweet Child",
    "Born To Run",
    "Dancing Queen",
    "Mr Brightside",
    "Take On Me",
    "Losing My Religion",
    "Black Dog",
    "Free Bird",
    "Brown Eyed Girl",
    "Tiny Dancer",
]
SONGS_EVAL = [
    "Let It Be",
    "Smooth Criminal",
    "Sweet Caroline",
    "Purple Rain",
    "Hotel California",
    "Comfortably Numb",
    "Karma Police",
    "Paint It Black",
]
TOPICS_TRAIN = [
    "the economy",
    "space exploration",
    "local elections",
    "climate policy",
    "technology",
    "the stock market",
    "public health",
    "renewable energy",
    "the labor market",
    "global trade",
    "education reform",
    "cybersecurity",
    "the music industry",
    "professional sports",
    "electric vehicles",
    "data privacy",
]
TOPICS_EVAL = [
    "artificial intelligence",
    "the housing market",
    "ocean conservation",
    "world cup",
    "gene editing",
    "supply chains",
    "the gig economy",
    "quantum computing",
]
# --- coding-agent surface (Claude Code / Codex-style tools) ---
PATHS_TRAIN = [
    "src/main.py",
    "utils/io.py",
    "app/server.js",
    "lib/parse.py",
    "tests/test_api.py",
    "README.md",
    "config/settings.yaml",
    "core/model.py",
    "src/train.py",
    "db/schema.sql",
    "src/utils/log.py",
    "api/handlers.go",
    "web/app.tsx",
    "scripts/deploy.sh",
    "models/encoder.py",
    "tests/test_db.py",
    "pkg/cache.rs",
    "cmd/root.go",
    "frontend/index.js",
    "data/clean.py",
    "services/auth.py",
    "internal/queue.go",
    "docs/setup.md",
    "build/make.sh",
    "ops/deploy.yaml",
    "src/cli.py",
    "lib/http_client.py",
]
PATHS_EVAL = [
    "data/loader.py",
    "web/index.html",
    "bin/run.sh",
    "docs/guide.md",
    "api/routes.go",
    "src/router.ts",
    "tests/test_cli.py",
    "pkg/store.rs",
    "config/prod.yaml",
    "services/email.py",
]
PATTERNS_TRAIN = [
    "TODO",
    "def main",
    "import os",
    "class Model",
    "async def",
    "API_KEY",
    "NotImplementedError",
    "print(",
    "import torch",
    "FIXME",
    "return self",
    "def __init__",
    "raise RuntimeError",
    "logging.info",
    "os.environ",
    "if __name__",
    "await fetch",
    "useState",
    "SELECT *",
    "panic(",
]
PATTERNS_EVAL = [
    "def run",
    "return None",
    "raise ValueError",
    "self.cfg",
    "import json",
    "class Config",
    "try:",
    "console.log",
]
COMMANDS_TRAIN = [
    "ls -la",
    "npm install",
    "pip install torch",
    "docker build .",
    "make test",
    "git status",
    "python -m pytest",
    "cargo run",
    "git diff",
    "npm run build",
    "kubectl get pods",
    "terraform plan",
    "go test ./...",
    "ruff check src",
    "docker compose up",
    "git log --oneline",
    "pip freeze",
    "yarn dev",
]
COMMANDS_EVAL = [
    "git pull",
    "cargo build",
    "python app.py",
    "npm run dev",
    "make lint",
    "docker ps",
    "go build ./...",
    "pytest -q",
]
COMMITS_TRAIN = [
    "fix bug",
    "add tests",
    "update docs",
    "refactor parser",
    "bump version",
    "improve logging",
    "fix typo",
    "add validation",
    "speed up query",
    "drop unused deps",
    "handle timeout",
    "rename module",
    "add CI step",
]
COMMITS_EVAL = [
    "handle edge case",
    "remove dead code",
    "tidy imports",
    "fix race condition",
    "add retry logic",
    "clarify error message",
]
TASKS_TRAIN = [
    "call the dentist",
    "buy groceries",
    "submit the report",
    "back up the laptop",
    "email the team",
    "renew the subscription",
    "pay the rent",
    "schedule a checkup",
    "reply to Sam",
    "order more coffee",
    "review the PR",
    "update the resume",
    "cancel the trial",
    "charge the camera",
]
TASKS_EVAL = [
    "water the plants",
    "renew the license",
    "book a flight",
    "return the package",
    "refill the prescription",
    "call the landlord",
]
DURATIONS_TRAIN = [
    "10 minutes",
    "1 hour",
    "30 seconds",
    "20 minutes",
    "2 hours",
    "15 minutes",
    "90 seconds",
    "4 hours",
    "25 minutes",
    "12 minutes",
    "6 hours",
    "40 seconds",
]
DURATIONS_EVAL = ["5 minutes", "45 seconds", "3 hours", "8 minutes", "75 minutes"]
# --- computer-use / productivity surface (calendar, email, browser, Notion, Slack, Jira) ---
URLS_TRAIN = [
    "example.com",
    "github.com",
    "wikipedia.org",
    "stackoverflow.com",
    "nytimes.com",
    "python.org",
    "arxiv.org",
    "docs.google.com",
    "amazon.com",
    "youtube.com",
    "linkedin.com",
    "medium.com",
    "bbc.com",
    "cnn.com",
    "apple.com",
    "microsoft.com",
    "gitlab.com",
    "npmjs.com",
    "pypi.org",
    "wikipedia.de",
]
URLS_EVAL = [
    "reddit.com",
    "figma.com",
    "openai.com",
    "huggingface.co",
    "spotify.com",
    "dropbox.com",
    "twitch.tv",
    "notion.so",
]
EVENT_TITLES_TRAIN = [
    "Team sync",
    "Standup",
    "Design review",
    "Budget meeting",
    "Onboarding",
    "Sprint review",
    "Project kickoff",
    "Retro",
    "All hands",
    "Lunch break",
    "Code freeze",
    "Release call",
    "Customer demo",
    "Strategy session",
    "Board meeting",
    "Coffee chat",
]
EVENT_TITLES_EVAL = ["Demo day", "Planning", "Interview", "Sync up", "Town hall", "Roadmap review"]
NOTION_TRAIN = [
    "meeting notes",
    "weekly goals",
    "project ideas",
    "reading list",
    "action items",
    "release plan",
    "design spec",
    "team updates",
    "research summary",
    "launch checklist",
    "interview feedback",
    "quarterly review",
]
NOTION_EVAL = ["bug triage", "roadmap draft", "retro notes", "onboarding guide", "budget plan"]
SLACK_TRAIN = [
    "deploy is done",
    "standup in 5",
    "PR is ready",
    "build passed",
    "reviewing now",
    "merging soon",
    "tests are green",
    "heading out",
    "back online",
    "looking into it",
    "good morning team",
    "rolling back",
]
SLACK_EVAL = ["ship it", "need help", "on my way", "almost there", "taking a break"]
JIRA_TRAIN = [
    "login bug",
    "slow query",
    "add dark mode",
    "fix typo",
    "memory leak",
    "flaky test",
    "improve search",
    "add pagination",
    "fix crash",
    "update schema",
    "rate limiting",
    "export to csv",
    "mobile layout",
    "session timeout",
]
JIRA_EVAL = ["broken link", "update deps", "cache miss", "form validation", "data loss", "404 page"]

# --- computer-use surface. All values are quoted in the prompt so they ground as literal
# substrings (the byte model has no vision; targets are semantic element descriptions, not pixels).
# UI element targets (click/double_click/move_cursor). Train/eval disjoint. ---
UI_TARGETS_TRAIN = [
    "the Submit button",
    "the Login button",
    "the Save icon",
    "the Search box",
    "the Settings menu",
    "the OK button",
    "the Cancel button",
    "the Next button",
    "the Profile tab",
    "the hamburger menu",
    "the Download link",
    "the Send button",
    "the Close button",
    "the Add button",
    "the Edit pencil",
    "the Filter dropdown",
    "the Share button",
    "the Upload area",
    "the Refresh icon",
    "the New Tab button",
    "the username field",
    "the password field",
    "the Continue button",
    "the dropdown arrow",
    "the checkbox",
    "the Reply button",
    "the Compose button",
]
UI_TARGETS_EVAL = [
    "the Delete button",
    "the Confirm button",
    "the Back arrow",
    "the email field",
    "the Sign Up button",
    "the Apply button",
    "the gear icon",
    "the notification bell",
    "the Done button",
    "the Bookmark star",
]
# text typed into fields (type_text / write_clipboard). Train/eval disjoint.
TYPED_TEXT_TRAIN = [
    "hello world",
    "admin@example.com",
    "my search query",
    "first last name",
    "the quick brown fox",
    "January 2026",
    "password123",
    "remote work policy",
    "quarterly report",
    "weekly standup notes",
    "order number 4521",
    "shipping address",
    "draft invoice",
    "project alpha",
    "status update",
]
TYPED_TEXT_EVAL = [
    "good afternoon",
    "test@mail.com",
    "annual budget",
    "release notes",
    "meeting agenda",
    "customer feedback",
    "invoice 9087",
]
# desktop apps (open_app). Train/eval disjoint.
APPS_TRAIN = [
    "Chrome",
    "Safari",
    "Slack",
    "Spotify",
    "Terminal",
    "VS Code",
    "Notion",
    "Figma",
    "Calculator",
    "Calendar",
    "Mail",
    "Photoshop",
    "Finder",
    "Discord",
    "Zoom",
    "Excel",
    "Word",
    "Preview",
    "Notes",
    "Messages",
]
APPS_EVAL = [
    "Firefox",
    "Postman",
    "Docker Desktop",
    "iTerm",
    "Obsidian",
    "Outlook",
    "Telegram",
    "PowerPoint",
]
# wait durations in seconds (wait.seconds integer extractor). Train/eval disjoint.
WAIT_SECONDS_TRAIN = [2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]
WAIT_SECONDS_EVAL = [1, 4, 8, 12, 25, 40, 75]

# --- modern dev / agentic surface ---
# Python snippets (run_python) — quoted so they ground exactly. Train/eval disjoint.
PYCODE_TRAIN = [
    "print('hi')",
    "import numpy as np",
    "x = sum(range(10))",
    "len(data)",
    "df.head()",
    "os.getcwd()",
    "2 ** 16",
    "sorted(items)",
    "model.eval()",
    "json.loads(s)",
    "np.zeros(5)",
    "open('f.txt').read()",
    "time.sleep(1)",
    "requests.get(url)",
    "plt.show()",
]
PYCODE_EVAL = [
    "print('done')",
    "math.sqrt(2)",
    "list(map(str, xs))",
    "pd.read_csv('a.csv')",
    "random.seed(0)",
    "min(values)",
    "torch.randn(3)",
]
# glob patterns (find_files) — quoted. Train/eval disjoint.
GLOBS_TRAIN = [
    "*.py",
    "**/*.js",
    "test_*.py",
    "*.log",
    "src/**/*.ts",
    "*.yaml",
    "Dockerfile",
    "*.md",
    "**/*.go",
    "*.csv",
    "config/*.json",
    "*.rs",
    "*.sql",
    "*.png",
]
GLOBS_EVAL = ["*.txt", "**/*.tsx", "*.toml", "spec_*.rb", "*.html", "lib/**/*.py", "*.env"]
# SQL queries (sql_query) — quoted. Train/eval disjoint.
SQL_TRAIN = [
    "SELECT * FROM users",
    "SELECT count(*) FROM orders",
    "DELETE FROM logs",
    "SELECT name FROM products",
    "UPDATE users SET active = 1",
    "SELECT * FROM events",
    "INSERT INTO tags VALUES (1)",
    "SELECT id FROM sessions",
    "SELECT * FROM payments",
    "SELECT email FROM accounts",
    "DROP TABLE temp",
    "SELECT * FROM inventory",
]
SQL_EVAL = [
    "SELECT * FROM customers",
    "SELECT max(price) FROM items",
    "TRUNCATE cache",
    "SELECT title FROM posts",
    "UPDATE orders SET paid = 1",
    "SELECT * FROM metrics",
]
# package names (install_package) — quoted. Train/eval disjoint.
PACKAGES_TRAIN = [
    "numpy",
    "pandas",
    "requests",
    "flask",
    "pytest",
    "torch",
    "fastapi",
    "redis",
    "django",
    "scipy",
    "click",
    "pydantic",
    "uvicorn",
    "aiohttp",
    "sqlalchemy",
    "matplotlib",
    "pillow",
    "boto3",
    "celery",
    "jinja2",
]
PACKAGES_EVAL = ["scikit-learn", "transformers", "httpx", "rich", "typer", "polars", "ruff", "lxml"]
# process names (kill_process) — quoted. Train/eval disjoint.
PROCESSES_TRAIN = [
    "node",
    "python",
    "chrome",
    "nginx",
    "postgres",
    "redis-server",
    "java",
    "docker",
    "vite",
    "webpack",
    "gunicorn",
    "mysqld",
    "ssh",
    "code",
]
PROCESSES_EVAL = ["firefox", "rabbitmq", "mongod", "celery", "uvicorn", "elasticsearch"]
# env var names (env_get) — quoted. Train/eval disjoint.
ENVVARS_TRAIN = [
    "PATH",
    "HOME",
    "API_KEY",
    "DATABASE_URL",
    "PORT",
    "AWS_REGION",
    "NODE_ENV",
    "PYTHONPATH",
    "SECRET_KEY",
    "REDIS_URL",
    "LOG_LEVEL",
    "USER",
    "SHELL",
    "OPENAI_API_KEY",
    "DEBUG",
]
ENVVARS_EVAL = ["LANG", "TZ", "GITHUB_TOKEN", "S3_BUCKET", "JAVA_HOME", "VIRTUAL_ENV", "EDITOR"]
# docker images (docker_run) — quoted. Train/eval disjoint.
IMAGES_TRAIN = [
    "nginx",
    "postgres",
    "redis",
    "ubuntu",
    "python:3.12",
    "node:20",
    "alpine",
    "mysql",
    "mongo",
    "busybox",
    "golang",
    "rust",
    "httpd",
    "memcached",
]
IMAGES_EVAL = ["debian", "elasticsearch", "rabbitmq", "grafana", "prometheus", "traefik"]
# archive paths (unzip) — file paths. Train/eval disjoint.
ARCHIVES_TRAIN = [
    "data.zip",
    "release.zip",
    "assets.zip",
    "backup.tar.gz",
    "dataset.zip",
    "logs.zip",
    "build/output.zip",
    "models.zip",
    "images.tar.gz",
    "dist.zip",
    "src/bundle.zip",
    "exports.zip",
]
ARCHIVES_EVAL = ["archive.zip", "vendor.tar.gz", "photos.zip", "downloads/pack.zip", "weights.zip"]

# --- implicit factual-question entities -> web_search. The query arg is grounded to the ENTITY,
# which is a literal substring of every templated factual question. Train/eval disjoint. ---
ENTITIES_TRAIN = [
    "Mount Everest",
    "the Eiffel Tower",
    "the Great Wall of China",
    "Lake Baikal",
    "the Amazon River",
    "the Sahara Desert",
    "the Pacific Ocean",
    "Mount Fuji",
    "the Nile",
    "the Colosseum",
    "the Empire State Building",
    "the Golden Gate Bridge",
    "the Grand Canyon",
    "Niagara Falls",
    "the Burj Khalifa",
    "the Statue of Liberty",
    "the Leaning Tower of Pisa",
    "the Sydney Opera House",
    "Mount Kilimanjaro",
    "the Mississippi River",
    "the Andes",
    "the Dead Sea",
    "the Sahara",
    "Big Ben",
    "the Hoover Dam",
    "the Panama Canal",
    "the Taj Mahal",
    "Stonehenge",
    "the Suez Canal",
    "the Mariana Trench",
    "Mount Rushmore",
    "the Rocky Mountains",
]
ENTITIES_EVAL = [
    "the Matterhorn",
    "Lake Victoria",
    "the Yangtze River",
    "the Gobi Desert",
    "the Arctic Ocean",
    "Mount Etna",
    "the Danube",
    "the Acropolis",
    "the Chrysler Building",
    "the Brooklyn Bridge",
    "Angkor Wat",
    "the Petronas Towers",
]
# "What's the {ATTR} of {PLACE}?" factual questions -> web_search (query grounded to PLACE).
PLACES_TRAIN = [
    "Peru",
    "Mongolia",
    "Iceland",
    "Portugal",
    "Kenya",
    "Vietnam",
    "Norway",
    "Chile",
    "Morocco",
    "Nepal",
    "Bolivia",
    "Finland",
    "Ireland",
    "Greece",
    "Croatia",
    "Ghana",
    "Ecuador",
    "Sweden",
    "Romania",
    "Hungary",
    "Tunisia",
    "Jordan",
    "Latvia",
    "Senegal",
    "Uruguay",
    "Slovenia",
    "Estonia",
    "Armenia",
    "Georgia",
    "Cambodia",
    "Laos",
    "Oman",
]
PLACES_EVAL = [
    "Paraguay",
    "Lithuania",
    "Slovakia",
    "Namibia",
    "Botswana",
    "Bhutan",
    "Moldova",
    "Albania",
    "Tanzania",
    "Zambia",
    "Belize",
    "Brunei",
]
# "Who {invented/wrote/founded/discovered} {THING}?" -> web_search (query grounded to THING).
INVENTIONS_TRAIN = [
    "the telephone",
    "the light bulb",
    "the printing press",
    "the steam engine",
    "the airplane",
    "the telescope",
    "the World Wide Web",
    "the radio",
    "the periodic table",
    "the theory of relativity",
    "the polio vaccine",
    "the cotton gin",
    "the sewing machine",
    "the microscope",
    "dynamite",
    "the assembly line",
    "the transistor",
    "penicillin",
    "the barometer",
    "Hamlet",
    "Moby Dick",
    "War and Peace",
    "the Mona Lisa",
    "Pride and Prejudice",
]
INVENTIONS_EVAL = [
    "the camera",
    "the typewriter",
    "the thermometer",
    "the compass",
    "the seismograph",
    "Frankenstein",
    "Don Quixote",
    "the Starry Night",
]
# "When did {EVENT} happen?" / "What year was {EVENT}?" -> web_search (query grounded to EVENT).
EVENTS_TRAIN = [
    "World War II",
    "the French Revolution",
    "the moon landing",
    "the Renaissance",
    "the Industrial Revolution",
    "the fall of the Berlin Wall",
    "the Cold War",
    "the American Civil War",
    "the Great Depression",
    "the Roman Empire",
    "the Battle of Hastings",
    "the Boston Tea Party",
    "the gold rush",
    "the Cuban Missile Crisis",
    "the signing of the Magna Carta",
    "the Black Death",
]
EVENTS_EVAL = [
    "the Spanish Inquisition",
    "the Wright brothers' first flight",
    "the Boston Marathon",
    "the invention of the internet",
    "the discovery of America",
    "the Russian Revolution",
]


def _slot_value_key(value: object) -> tuple[str, object]:
    """Normalize named slot values for conservative train/eval collision checks."""

    if isinstance(value, str):
        return ("string", _canonical_holdout_prompt(value))
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("numeric", float(value))
    return (type(value).__name__, repr(value))


def _paper_train_v2_slot_audit() -> dict[str, object]:
    """Fail if legacy named pools or v2 train-only slots collide with frozen eval slots."""

    namespace = globals()
    paired_pools = 0
    frozen_eval_values: set[tuple[str, object]] = set()
    for name, values in sorted(namespace.items()):
        if not name.endswith("_EVAL"):
            continue
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"frozen eval slot pool {name} must be a non-empty sequence")
        frozen_eval_values.update(_slot_value_key(value) for value in values)

    for name, train_values in sorted(namespace.items()):
        if not name.endswith("_TRAIN"):
            continue
        eval_name = f"{name[:-6]}_EVAL"
        if eval_name not in namespace:
            continue
        eval_values = namespace[eval_name]
        if not isinstance(train_values, (list, tuple)) or not train_values:
            raise ValueError(f"training slot pool {name} must be a non-empty sequence")
        train_keys = {_slot_value_key(value) for value in train_values}
        eval_keys = {_slot_value_key(value) for value in eval_values}
        overlap = sorted(train_keys & eval_keys, key=repr)
        if overlap:
            raise ValueError(
                f"train/eval slot pool collision for {name}/{eval_name}: {overlap[0]!r}"
            )
        paired_pools += 1

    v2_pool_counts: dict[str, int] = {}
    for name, values in sorted(PAPER_TRAIN_V2_SLOT_POOLS.items()):
        if not values:
            raise ValueError(f"paper_train_v2 slot pool {name} must not be empty")
        train_keys = {_slot_value_key(value) for value in values}
        overlap = sorted(train_keys & frozen_eval_values, key=repr)
        if overlap:
            raise ValueError(
                f"paper_train_v2 slot pool {name} collides with frozen eval slots: "
                f"{overlap[0]!r}"
            )
        v2_pool_counts[name] = len(train_keys)

    return {
        "normalization": "typed values; strings use Unicode NFKC, whitespace collapse, casefold",
        "paired_legacy_train_eval_pools": paired_pools,
        "frozen_eval_atomic_values": len(frozen_eval_values),
        "paper_v2_train_only_pool_values": v2_pool_counts,
        "overlap": 0,
        "boolean_domain_note": (
            "true/false are closed schema primitives, not split-specific named slot values"
        ),
    }


# A realistic usage distribution (not the old calc-dominated one): emphasize the two-call
# ("parallel") turns and productivity tools people actually want; down-weight the over-represented
# calculator/weather. Used as the base sampling weight in the flywheels.
REALISTIC_WEIGHTS = {
    "parallel": 2.5,
    "calc": 0.3,
    "weather": 0.6,
    "send_email": 1.4,
    "calendar_event": 1.4,
    "open_url": 1.4,
    "slack_send": 1.4,
    "notion_write": 1.4,
    "jira_issue": 1.4,
    "set_reminder": 1.2,
    "set_timer": 1.2,
    "read_file": 1.2,
    "write_file": 1.2,
    "run_command": 1.2,
    "git_commit": 1.2,
    # computer-use family — the headline new capability, weight it up
    "click": 1.6,
    "type_text": 1.5,
    "key_press": 1.3,
    "scroll": 1.2,
    "screenshot": 1.1,
    "double_click": 1.2,
    "drag": 1.2,
    "wait": 1.0,
    "move_cursor": 1.1,
    "open_app": 1.5,
    # modern dev / agentic tools
    "run_python": 1.4,
    "edit_file": 1.3,
    "apply_patch": 1.1,
    "http_request": 1.3,
    "sql_query": 1.3,
    "list_dir": 1.2,
    "find_files": 1.2,
    "git_diff": 1.1,
    "git_status": 1.1,
    "install_package": 1.3,
    "kill_process": 1.1,
    "read_clipboard": 0.9,
    "write_clipboard": 1.0,
    "download_file": 1.2,
    "unzip": 1.0,
    "env_get": 1.0,
    "make_dir": 1.0,
    "list_processes": 0.9,
    "docker_run": 1.2,
}


@dataclass
class Sample:
    category: str  # weather | calc | ... | parallel | text | no_tool
    group: str  # tool_call | web_search | planner | ... | parallel | text
    prompt: str  # user text (without framing markers)
    kind: str  # "tool" or "text"
    target: str  # canonical assistant body (no markers/EOS)
    ref_name: str = ""  # tool name (for tool kind; first call for parallel)
    ref_args: str = ""  # canonical sorted-key JSON args string (first call)
    calls: list = None  # [{"name","arguments"}, ...] when >1 call (parallel); else None


def _tool_target(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"), sort_keys=True)


class Generator:
    _LEGACY_MODE = "legacy_v1"
    _EVAL_EPISODE_WRAPPERS = (
        "Could you help with this request? {request}",
        "Please take care of this: {request}",
        "Here's what I need: {request}",
    )
    _PAPER_V2_SCROLL_DIRECTIONS = ("up", "down", "left", "right")
    _PAPER_V2_SCROLL_TEMPLATE_COUNT = 6
    _PAPER_V2_NO_TOOL_TEMPLATE_COUNT = 8
    _PAPER_V2_SCHEMA_EPISODE_TEMPLATE_COUNT = 4
    _PAPER_V2_RECOVERY_EPISODE_TEMPLATE_COUNT = 3

    def __init__(
        self,
        level: int = 1,
        seed: int = 0,
        split: str = "train",
        mode: str = _LEGACY_MODE,
    ):
        if mode not in {self._LEGACY_MODE, PAPER_TRAIN_V2_MODE}:
            raise ValueError(f"unsupported deterministic generator mode: {mode!r}")
        if mode == PAPER_TRAIN_V2_MODE and split != "train":
            raise ValueError("paper_train_v2 is train-only; frozen evaluation rows must stay v1")
        self.level = level
        self.split = split
        self.mode = mode
        self.rng = random.Random(seed)
        tr = split == "train"
        self.cities = CITIES_TRAIN if tr else CITIES_EVAL
        self.names = NAMES_TRAIN if tr else NAMES_EVAL
        self.queries = QUERIES_TRAIN if tr else QUERIES_EVAL
        self.goals = GOALS_TRAIN if tr else GOALS_EVAL
        self.terms = TERMS_TRAIN if tr else TERMS_EVAL
        self.songs = SONGS_TRAIN if tr else SONGS_EVAL
        self.topics = TOPICS_TRAIN if tr else TOPICS_EVAL
        self.paths = PATHS_TRAIN if tr else PATHS_EVAL
        self.patterns = PATTERNS_TRAIN if tr else PATTERNS_EVAL
        self.commands = COMMANDS_TRAIN if tr else COMMANDS_EVAL
        self.commits = COMMITS_TRAIN if tr else COMMITS_EVAL
        self.tasks = TASKS_TRAIN if tr else TASKS_EVAL
        self.durations = DURATIONS_TRAIN if tr else DURATIONS_EVAL
        self.urls = URLS_TRAIN if tr else URLS_EVAL
        self.events = EVENT_TITLES_TRAIN if tr else EVENT_TITLES_EVAL
        self.notion = NOTION_TRAIN if tr else NOTION_EVAL
        self.slack = SLACK_TRAIN if tr else SLACK_EVAL
        self.jira = JIRA_TRAIN if tr else JIRA_EVAL
        self.entities = ENTITIES_TRAIN if tr else ENTITIES_EVAL
        self.places = PLACES_TRAIN if tr else PLACES_EVAL
        self.inventions = INVENTIONS_TRAIN if tr else INVENTIONS_EVAL
        self.history = EVENTS_TRAIN if tr else EVENTS_EVAL  # historical events for web_search
        # computer-use surface
        self.ui_targets = UI_TARGETS_TRAIN if tr else UI_TARGETS_EVAL
        self.typed_text = TYPED_TEXT_TRAIN if tr else TYPED_TEXT_EVAL
        self.apps = APPS_TRAIN if tr else APPS_EVAL
        self.wait_seconds = WAIT_SECONDS_TRAIN if tr else WAIT_SECONDS_EVAL
        # modern dev / agentic surface
        self.pycode = PYCODE_TRAIN if tr else PYCODE_EVAL
        self.globs = GLOBS_TRAIN if tr else GLOBS_EVAL
        self.sql = SQL_TRAIN if tr else SQL_EVAL
        self.packages = PACKAGES_TRAIN if tr else PACKAGES_EVAL
        self.processes = PROCESSES_TRAIN if tr else PROCESSES_EVAL
        self.envvars = ENVVARS_TRAIN if tr else ENVVARS_EVAL
        self.images = IMAGES_TRAIN if tr else IMAGES_EVAL
        self.archives = ARCHIVES_TRAIN if tr else ARCHIVES_EVAL

    def _split_choice(self, train_values, eval_values):
        """Choose from a phrasing/value pool reserved for this data split."""

        return self.rng.choice(train_values if self.split == "train" else eval_values)

    # ---- per-category sample makers ----
    def weather(self) -> Sample:
        city = self.rng.choice(self.cities)
        phr = self.rng.choice(
            [
                f"What's the weather in {city}?",
                f"Tell me the weather for {city}.",
                f"How is the weather in {city} right now?",
                f"I wonder what the weather's like in {city}.",
                f"Is it raining in {city}?",
                f"What's it like outside in {city}?",
            ]
            + ([f"Weather in {city} please."] if self.level >= 4 else [])
        )
        args = {"city": city}
        if self.level >= 2 and self.rng.random() < 0.5:
            unit = self.rng.choice(["c", "f"])
            args["unit"] = unit
            phr += f" In {'Celsius' if unit == 'c' else 'Fahrenheit'}."
        return Sample(
            "weather",
            "tool_call",
            phr,
            "tool",
            _tool_target("get_weather", args),
            "get_weather",
            json.dumps(args, separators=(",", ":"), sort_keys=True),
        )

    def calc(self) -> Sample:
        operands = range(1, 21) if self.split == "train" else range(21, 41)
        a, b = self.rng.choice(operands), self.rng.choice(operands)
        op = self.rng.choice(["+", "-", "*"])
        if self.level >= 3 and self.rng.random() < 0.5:
            c = self.rng.choice(operands)
            op2 = self.rng.choice(["+", "-", "*"])
            expr = f"{a}{op}{b}{op2}{c}"
            q = f"What is {a} {op} {b} {op2} {c}?"
        else:
            expr = f"{a}{op}{b}"
            q = f"What is {a} {op} {b}?"
        # Natural arithmetic phrasings -> calculator. The arithmetic schema extractor grounds the
        # expression from the digit/operator span (whitespace-stripped), so we keep the SYMBOL form
        # (e.g. "7*8") inside the prompt rather than word forms ("7 times 8") that wouldn't ground.
        if self.rng.random() < 0.5 and expr == f"{a}{op}{b}":
            q = self.rng.choice(
                [
                    f"How much is {a}{op}{b}?",
                    f"Can you compute {a}{op}{b}?",
                    f"What's {a}{op}{b}?",
                    f"Calculate {a}{op}{b}.",
                    f"Work out {a}{op}{b} for me.",
                ]
            )
        args = {"expression": expr}
        return Sample(
            "calc",
            "tool_call",
            q,
            "tool",
            _tool_target("calculator", args),
            "calculator",
            json.dumps(args, separators=(",", ":"), sort_keys=True),
        )

    def web_search(self) -> Sample:
        query = self.rng.choice(self.queries)
        phr = self.rng.choice(
            [
                f"Search the web for {query}.",
                f"Look up {query} online.",
                f"Find information about {query}.",
                f"Can you look up {query}?",
                f"I'm searching for {query}.",
                f"Search for {query}.",
            ]
        )
        args = {"query": query}
        return Sample(
            "web_search",
            "web_search",
            phr,
            "tool",
            _tool_target("web_search", args),
            "web_search",
            json.dumps(args, separators=(",", ":"), sort_keys=True),
        )

    def web_search_implicit(self) -> Sample:
        """Implicit factual questions (bare questions, NOT explicit "search" commands) that should
        still map to ``web_search``. The ``query`` arg is grounded to an ENTITY/PLACE/THING/EVENT
        that is a literal substring of the chosen phrasing. This is the headline coverage gap: the
        model previously only saw imperative "search for X" prompts and mis-routed real questions."""
        kind = self.rng.choice(["measure", "attr", "who", "when"])
        if kind == "measure":
            e = self.rng.choice(self.entities)
            adj = self.rng.choice(
                ["tall", "high", "far away", "old", "long", "deep", "heavy", "big", "wide"]
            )
            phr = self.rng.choice(
                [
                    f"How {adj} is {e}?",
                    f"Do you know how {adj} {e} is?",
                    f"I wonder how {adj} {e} is.",
                ]
            )
            query = e
        elif kind == "attr":
            p = self.rng.choice(self.places)
            attr = self.rng.choice(
                [
                    "capital",
                    "population",
                    "currency",
                    "national language",
                    "area",
                    "time zone",
                    "flag",
                ]
            )
            phr = self.rng.choice(
                [
                    f"What's the {attr} of {p}?",
                    f"What is the {attr} of {p}?",
                    f"Tell me the {attr} of {p}.",
                    f"Do you know the {attr} of {p}?",
                ]
            )
            query = p
        elif kind == "who":
            t = self.rng.choice(self.inventions)
            verb = self.rng.choice(
                [
                    "invented",
                    "wrote",
                    "founded",
                    "discovered",
                    "designed",
                    "built",
                    "painted",
                    "created",
                ]
            )
            phr = self.rng.choice(
                [
                    f"Who {verb} {t}?",
                    f"Do you know who {verb} {t}?",
                    f"Any idea who {verb} {t}?",
                ]
            )
            query = t
        else:  # when
            ev = self.rng.choice(self.history)
            phr = self.rng.choice(
                [
                    f"When did {ev} happen?",
                    f"What year was {ev}?",
                    f"When was {ev}?",
                    f"What year did {ev} take place?",
                ]
            )
            query = ev
        args = {"query": query}
        return Sample(
            "web_search",
            "web_search",
            phr,
            "tool",
            _tool_target("web_search", args),
            "web_search",
            json.dumps(args, separators=(",", ":"), sort_keys=True),
        )

    def planner(self) -> Sample:
        goal = self.rng.choice(self.goals)
        phr = self.rng.choice(
            [
                f"Make a plan to {goal}.",
                f"I want to {goal}.",
                f"I need to {goal}.",
                f"My goal is to {goal}.",
                f"Create a plan to {goal}.",
                f"Plan how to {goal}.",
            ]
        )
        args = {"goal": goal}
        return Sample(
            "planner",
            "planner",
            phr,
            "tool",
            _tool_target("planner", args),
            "planner",
            json.dumps(args, separators=(",", ":"), sort_keys=True),
        )

    def _string_tool(self, category, group, name, arg, value, phrasings) -> Sample:
        args = {arg: value}
        return Sample(
            category,
            group,
            self.rng.choice(phrasings),
            "tool",
            _tool_target(name, args),
            name,
            json.dumps(args, separators=(",", ":"), sort_keys=True),
        )

    def define(self) -> Sample:
        t = self.rng.choice(self.terms)
        # Meaning questions -> define (distinct from web_search facts and get_news current events).
        return self._string_tool(
            "define",
            "define",
            "define",
            "term",
            t,
            [
                f"Definition of {t}.",
                f"Define {t}.",
                f"Explain {t}.",
                f"Tell me about {t}.",
                f"Describe {t}.",
                f"Give me the definition of {t}.",
                f"What does {t} mean?",
                f"What is the meaning of {t}?",
                f"What's the definition of {t}?",
                f"What does the word {t} mean?",
                f"Can you define {t}?",
            ],
        )

    def play_music(self) -> Sample:
        s = self.rng.choice(self.songs)
        return self._string_tool(
            "play_music",
            "music",
            "play_music",
            "song",
            s,
            [
                f"Play {s}.",
                f"Put on {s}.",
                f"Start playing {s}.",
                f"Queue up {s}.",
                f"I want to hear {s}.",
                f"Play the song {s}.",
                f"Can you put on {s}?",
                f"I'm in the mood for {s}.",
                f"Let's listen to {s}.",
            ],
        )

    def get_news(self) -> Sample:
        t = self.rng.choice(self.topics)
        # Current-events questions -> get_news (distinct from define meaning & web_search facts).
        return self._string_tool(
            "get_news",
            "news",
            "get_news",
            "topic",
            t,
            [
                f"Show the news about {t}.",
                f"Latest news on {t}.",
                f"What's the news about {t}.",
                f"Any news about {t}?",
                f"Give me news on {t}.",
                f"Show news about {t}.",
                f"What's the latest on {t}?",
                f"What's happening with {t}?",
                f"Any updates on {t}?",
                f"What's new with {t}?",
                f"Catch me up on {t}.",
            ],
        )

    # --- coding-agent tools (Claude Code / Codex-style) ---
    def read_file(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool(
            "read_file",
            "code",
            "read_file",
            "path",
            p,
            [
                f"Read the file {p}.",
                f"Open {p}.",
                f"Show the contents of {p}.",
                f"Display {p}.",
                f"Cat {p}.",
                f"Show me {p}.",
            ],
        )

    def write_file(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool(
            "write_file",
            "code",
            "write_file",
            "path",
            p,
            [
                f"Create the file {p}.",
                f"Write to {p}.",
                f"Save the file {p}.",
                f"Add a new file {p}.",
            ],
        )

    def grep_search(self) -> Sample:
        pat = self.rng.choice(self.patterns)
        return self._string_tool(
            "grep_search",
            "code",
            "grep_search",
            "pattern",
            pat,
            [
                f"Search the code for '{pat}'.",
                f"Grep for '{pat}'.",
                f"Find '{pat}' in the repo.",
                f"Where is '{pat}' used?",
                f"Locate '{pat}' in the code.",
                f"Search for '{pat}'.",
            ],
        )

    def run_command(self) -> Sample:
        c = self.rng.choice(self.commands)
        return self._string_tool(
            "run_command",
            "code",
            "run_command",
            "command",
            c,
            [
                f"Run the command '{c}'.",
                f"Execute '{c}' in the shell.",
                f"Run '{c}'.",
                f"Execute the command '{c}'.",
            ],
        )

    def git_commit(self) -> Sample:
        msg = self.rng.choice(self.commits)
        return self._string_tool(
            "git_commit",
            "code",
            "git_commit",
            "message",
            msg,
            [
                f"Commit with message '{msg}'.",
                f"Make a git commit saying '{msg}'.",
                f"Git commit with '{msg}'.",
                f"Commit the changes: '{msg}'.",
            ],
        )

    def run_tests(self) -> Sample:
        phr = self._split_choice(
            ["Run the tests.", "Run the test suite.", "Execute all tests."],
            [
                "Could you run the tests?",
                "Please execute the test suite.",
                "Run all tests now.",
            ],
        )
        return Sample(
            "run_tests", "code", phr, "tool", _tool_target("run_tests", {}), "run_tests", "{}"
        )

    # --- popular everyday tools ---
    def set_reminder(self) -> Sample:
        t = self.rng.choice(self.tasks)
        return self._string_tool(
            "set_reminder",
            "tool_call",
            "set_reminder",
            "task",
            t,
            [
                f"Set a reminder to {t}.",
                f"Remind to {t}.",
                f"Remind me to {t}.",
                f"Add a reminder to {t}.",
                f"Can you remind me to {t}?",
                f"Don't let me forget to {t}.",
                f"I need to remember to {t}.",
            ],
        )

    def set_timer(self) -> Sample:
        d = self.rng.choice(self.durations)
        return self._string_tool(
            "set_timer",
            "tool_call",
            "set_timer",
            "duration",
            d,
            [
                f"Set a timer for {d}.",
                f"Start a timer for {d}.",
                f"Set a countdown for {d}.",
                f"Wake me in {d}.",
                f"Can you set a timer for {d}?",
                f"Let me know in {d}.",
                f"Ping me in {d}.",
            ],
        )

    # --- computer-use / productivity tools ---
    def calendar_event(self) -> Sample:
        t = self.rng.choice(self.events)
        return self._string_tool(
            "calendar_event",
            "productivity",
            "calendar_event",
            "title",
            t,
            [
                f"Add a calendar event called '{t}'.",
                f"Schedule '{t}' on my calendar.",
                f"Create a calendar event '{t}'.",
                f"Put '{t}' on the calendar.",
            ],
        )

    def send_email(self) -> Sample:
        nm = self.rng.choice(self.names)
        return self._string_tool(
            "send_email",
            "productivity",
            "send_email",
            "recipient",
            nm,
            [
                f"Send an email to {nm}.",
                f"Email {nm}.",
                f"Write an email to {nm}.",
                f"Compose an email to {nm}.",
                f"Can you email {nm}?",
                f"Shoot an email over to {nm}.",
                f"Drop {nm} an email.",
            ],
        )

    def open_url(self) -> Sample:
        u = self.rng.choice(self.urls)
        return self._string_tool(
            "open_url",
            "browser",
            "open_url",
            "url",
            u,
            [
                f"Open {u}.",
                f"Go to {u}.",
                f"Navigate to {u} in the browser.",
                f"Visit {u}.",
                f"Pull up {u}.",
            ],
        )

    def notion_write(self) -> Sample:
        c = self.rng.choice(self.notion)
        return self._string_tool(
            "notion_write",
            "productivity",
            "notion_write",
            "content",
            c,
            [
                f"Write '{c}' in Notion.",
                f"Add a Notion note saying '{c}'.",
                f"Note '{c}' in Notion.",
                f"Save '{c}' to Notion.",
            ],
        )

    def slack_send(self) -> Sample:
        m = self.rng.choice(self.slack)
        return self._string_tool(
            "slack_send",
            "productivity",
            "slack_send",
            "message",
            m,
            [
                f"Send a Slack message saying '{m}'.",
                f"Post '{m}' to Slack.",
                f"Slack the team '{m}'.",
                f"Send '{m}' on Slack.",
            ],
        )

    def jira_issue(self) -> Sample:
        s = self.rng.choice(self.jira)
        return self._string_tool(
            "jira_issue",
            "productivity",
            "jira_issue",
            "summary",
            s,
            [
                f"Create a Jira ticket titled '{s}'.",
                f"Open a Jira issue for '{s}'.",
                f"File a Jira bug for '{s}'.",
                f"Log a Jira issue: '{s}'.",
            ],
        )

    # --- computer-use family (text-grounded GUI actions) -------------------------------------
    # All string args are wrapped in single quotes inside the prompt so the `quoted` extractor
    # grounds them as exact substrings (the byte model has no vision — targets are semantic
    # element descriptions, never pixel coordinates). Enums/ints ground via schema extractors.
    def _tool_sample(self, category, group, name, args) -> Sample:
        ref_args = json.dumps(args, separators=(",", ":"), sort_keys=True)
        return Sample(category, group, "", "tool", _tool_target(name, args), name, ref_args)

    def screenshot(self) -> Sample:
        phr = self._split_choice(
            [
                "Take a screenshot.",
                "Capture the screen.",
                "Grab a screenshot of the screen.",
                "Screenshot the current screen.",
                "Snap a picture of what's on screen.",
            ],
            [
                "Could you take a screenshot?",
                "Snap the screen.",
                "Capture what's on screen.",
                "Get a screenshot.",
                "Take a picture of the screen.",
            ],
        )
        s = self._tool_sample("screenshot", "computer_use", "screenshot", {})
        s.prompt = phr
        return s

    def click(self) -> Sample:
        t = self.rng.choice(self.ui_targets)
        s = self._tool_sample("click", "computer_use", "click", {"target": t})
        s.prompt = self.rng.choice(
            [
                f"Click '{t}'.",
                f"Click on '{t}'.",
                f"Press '{t}'.",
                f"Tap '{t}'.",
                f"Hit '{t}'.",
                f"Select '{t}'.",
                f"Go ahead and click '{t}'.",
            ]
        )
        return s

    def double_click(self) -> Sample:
        t = self.rng.choice(self.ui_targets)
        s = self._tool_sample("double_click", "computer_use", "double_click", {"target": t})
        s.prompt = self.rng.choice(
            [
                f"Double-click '{t}'.",
                f"Double click '{t}'.",
                f"Double-click on '{t}'.",
                f"Open '{t}' by double-clicking.",
                f"Double tap '{t}'.",
            ]
        )
        return s

    def type_text(self) -> Sample:
        txt = self.rng.choice(self.typed_text)
        s = self._tool_sample("type_text", "computer_use", "type_text", {"text": txt})
        s.prompt = self.rng.choice(
            [
                f"Type '{txt}'.",
                f"Type '{txt}' into the field.",
                f"Enter '{txt}'.",
                f"Input '{txt}'.",
                f"Write '{txt}' in the box.",
                f"Fill in '{txt}'.",
            ]
        )
        return s

    def key_press(self) -> Sample:
        key = self.rng.choice(
            [
                "Enter",
                "Tab",
                "Escape",
                "Backspace",
                "Space",
                "Delete",
                "ArrowUp",
                "ArrowDown",
                "ArrowLeft",
                "ArrowRight",
            ]
        )
        s = self._tool_sample("key_press", "computer_use", "key_press", {"key": key})
        s.prompt = self._split_choice(
            [
                f"Press {key}.",
                f"Hit the {key} key.",
                f"Press the {key} key.",
                f"Tap {key}.",
                f"Send a {key} keypress.",
            ],
            [
                f"Could you press {key}?",
                f"Push the {key} key.",
                f"Trigger {key}.",
                f"Press down {key}.",
                f"Strike the {key} key.",
            ],
        )
        return s

    def scroll(self) -> Sample:
        d = self.rng.choice(["up", "down", "left", "right"])
        s = self._tool_sample("scroll", "computer_use", "scroll", {"direction": d})
        s.prompt = self._split_choice(
            [
                f"Scroll {d}.",
                f"Scroll {d} a bit.",
                f"Scroll the page {d}.",
                f"Please scroll {d}.",
                f"Keep scrolling {d}.",
            ],
            [
                f"Could you scroll {d}?",
                f"Scroll a little {d}.",
                f"Nudge the page {d}.",
                f"Scroll further {d}.",
                f"Pan {d}.",
            ],
        )
        return s

    def precise_scroll_v2(self) -> Sample:
        """Paper-v2-only scroll call covering JSON number and boolean arguments together."""

        if self.mode != PAPER_TRAIN_V2_MODE:
            raise ValueError("precise_scroll_v2 is available only in paper_train_v2 mode")
        direction = self.rng.choice(self._PAPER_V2_SCROLL_DIRECTIONS)
        amount = self.rng.choice(PAPER_V2_SCROLL_AMOUNTS_TRAIN)
        cue, smooth = self.rng.choice(PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN)
        args = {"direction": direction, "amount": amount, "smooth": smooth}
        sample = self._tool_sample(
            "precise_scroll",
            "computer_use_v2",
            "scroll",
            args,
        )
        sample.prompt = self.rng.choice(
            [
                f"Scroll {direction} by {amount} screen lengths and {cue}.",
                f"{cue.capitalize()}, then scroll {direction} exactly {amount} screen lengths.",
                f"Move the page {direction} {amount} screen lengths; {cue}.",
                f"Please scroll {direction} for {amount} screen lengths and {cue}.",
                f"Use a {amount}-screen-length scroll {direction}; {cue}.",
                f"{cue.capitalize()} and nudge the page {direction} by {amount} screen lengths.",
            ]
        )
        return sample

    def drag(self) -> Sample:
        src, dst = self.rng.sample(self.ui_targets, 2)
        s = self._tool_sample("drag", "computer_use", "drag", {"source": src, "dest": dst})
        s.prompt = self.rng.choice(
            [
                f"Drag '{src}' to '{dst}'.",
                f"Drag '{src}' onto '{dst}'.",
                f"Move '{src}' over to '{dst}' by dragging.",
                f"Drag and drop '{src}' to '{dst}'.",
            ]
        )
        return s

    def wait(self) -> Sample:
        sec = self.rng.choice(self.wait_seconds)
        s = self._tool_sample("wait", "computer_use", "wait", {"seconds": int(sec)})
        s.prompt = self.rng.choice(
            [
                f"Wait {sec} seconds.",
                f"Wait for {sec} seconds.",
                f"Pause for {sec} seconds.",
                f"Hold on {sec} seconds.",
                f"Give it {sec} seconds.",
            ]
        )
        return s

    def move_cursor(self) -> Sample:
        t = self.rng.choice(self.ui_targets)
        s = self._tool_sample("move_cursor", "computer_use", "move_cursor", {"target": t})
        s.prompt = self.rng.choice(
            [
                f"Move the cursor to '{t}'.",
                f"Hover over '{t}'.",
                f"Move the mouse to '{t}'.",
                f"Point at '{t}'.",
                f"Bring the cursor to '{t}'.",
            ]
        )
        return s

    def open_app(self) -> Sample:
        nm = self.rng.choice(self.apps)
        s = self._tool_sample("open_app", "computer_use", "open_app", {"name": nm})
        s.prompt = self.rng.choice(
            [
                f"Open '{nm}'.",
                f"Launch '{nm}'.",
                f"Open the '{nm}' app.",
                f"Start '{nm}'.",
                f"Fire up '{nm}'.",
                f"Bring up '{nm}'.",
            ]
        )
        return s

    # --- modern dev / agentic tools ----------------------------------------------------------
    def run_python(self) -> Sample:
        code = self.rng.choice(self.pycode)
        s = self._tool_sample("run_python", "code", "run_python", {"code": code})
        s.prompt = self.rng.choice(
            [
                f"Run the Python code '{code}'.",
                f"Execute '{code}' in Python.",
                f"Run '{code}'.",
                f"Evaluate '{code}' in a Python shell.",
            ]
        )
        return s

    def edit_file(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool(
            "edit_file",
            "code",
            "edit_file",
            "path",
            p,
            [
                f"Edit {p}.",
                f"Make changes to {p}.",
                f"Modify {p}.",
                f"Update the file {p}.",
                f"Open {p} for editing.",
            ],
        )

    def apply_patch(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool(
            "apply_patch",
            "code",
            "apply_patch",
            "path",
            p,
            [
                f"Apply the patch to {p}.",
                f"Patch {p}.",
                f"Apply a patch to the file {p}.",
                f"Patch the file {p}.",
            ],
        )

    def http_request(self) -> Sample:
        u = self.rng.choice(self.urls)
        args = {"url": u}
        if self.rng.random() < 0.5:
            method = self.rng.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])
            args["method"] = method
            phr = self.rng.choice(
                [
                    f"Make a {method} request to {u}.",
                    f"Send a {method} request to {u}.",
                    f"Do a {method} on {u}.",
                ]
            )
        else:
            phr = self.rng.choice(
                [
                    f"Make an HTTP request to {u}.",
                    f"Hit the endpoint {u}.",
                    f"Send a request to {u}.",
                    f"Call the API at {u}.",
                ]
            )
        s = self._tool_sample("http_request", "code", "http_request", args)
        s.prompt = phr
        return s

    def sql_query(self) -> Sample:
        q = self.rng.choice(self.sql)
        return self._string_tool(
            "sql_query",
            "code",
            "sql_query",
            "query",
            q,
            [
                f"Run the SQL query '{q}'.",
                f"Execute '{q}' on the database.",
                f"Query the database with '{q}'.",
                f"Run '{q}' against the db.",
            ],
        )

    def list_dir(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool(
            "list_dir",
            "code",
            "list_dir",
            "path",
            p,
            [
                f"List the directory {p}.",
                f"List the contents of {p}.",
                f"Show what's in {p}.",
                f"ls {p}.",
                f"What files are in {p}?",
            ],
        )

    def find_files(self) -> Sample:
        g = self.rng.choice(self.globs)
        return self._string_tool(
            "find_files",
            "code",
            "find_files",
            "pattern",
            g,
            [
                f"Find files matching '{g}'.",
                f"Find all '{g}' files.",
                f"Search for files matching '{g}'.",
                f"Locate '{g}' files.",
            ],
        )

    def git_diff(self) -> Sample:
        s = self._tool_sample("git_diff", "code", "git_diff", {})
        s.prompt = self._split_choice(
            [
                "Show the git diff.",
                "What's the diff?",
                "Git diff.",
                "Show me the changes.",
                "Display the current diff.",
            ],
            [
                "Could you show the diff?",
                "Run git diff.",
                "What's the current diff?",
                "Show me the unstaged changes.",
                "Diff the working tree.",
            ],
        )
        return s

    def git_status(self) -> Sample:
        s = self._tool_sample("git_status", "code", "git_status", {})
        s.prompt = self._split_choice(
            [
                "Show the git status.",
                "What's the git status?",
                "Git status.",
                "Check the repo status.",
                "Show me the working tree status.",
            ],
            [
                "Could you check git status?",
                "Run git status.",
                "What does git status say?",
                "Give me the repo status.",
                "Where does the repo stand?",
            ],
        )
        return s

    def install_package(self) -> Sample:
        nm = self.rng.choice(self.packages)
        return self._string_tool(
            "install_package",
            "code",
            "install_package",
            "name",
            nm,
            [
                f"Install '{nm}'.",
                f"Install the package '{nm}'.",
                f"Add the dependency '{nm}'.",
                f"pip install '{nm}'.",
                f"Set up '{nm}'.",
            ],
        )

    def kill_process(self) -> Sample:
        nm = self.rng.choice(self.processes)
        return self._string_tool(
            "kill_process",
            "code",
            "kill_process",
            "name",
            nm,
            [
                f"Kill the process '{nm}'.",
                f"Kill '{nm}'.",
                f"Stop the '{nm}' process.",
                f"Terminate '{nm}'.",
            ],
        )

    def read_clipboard(self) -> Sample:
        s = self._tool_sample("read_clipboard", "computer_use", "read_clipboard", {})
        s.prompt = self._split_choice(
            [
                "Read the clipboard.",
                "What's on the clipboard?",
                "Get the clipboard contents.",
                "Paste the clipboard.",
                "Show me what's copied.",
            ],
            [
                "Could you read the clipboard?",
                "What's copied?",
                "Grab the clipboard contents.",
                "Show the clipboard.",
                "Fetch the clipboard.",
            ],
        )
        return s

    def write_clipboard(self) -> Sample:
        txt = self.rng.choice(self.typed_text)
        return self._string_tool(
            "write_clipboard",
            "computer_use",
            "write_clipboard",
            "text",
            txt,
            [
                f"Copy '{txt}' to the clipboard.",
                f"Put '{txt}' on the clipboard.",
                f"Copy '{txt}'.",
                f"Set the clipboard to '{txt}'.",
            ],
        )

    def download_file(self) -> Sample:
        u = self.rng.choice(self.urls)
        return self._string_tool(
            "download_file",
            "code",
            "download_file",
            "url",
            u,
            [
                f"Download the file from {u}.",
                f"Download {u}.",
                f"Fetch the file at {u}.",
                f"Grab the file from {u}.",
            ],
        )

    def unzip(self) -> Sample:
        p = self.rng.choice(self.archives)
        return self._string_tool(
            "unzip",
            "code",
            "unzip",
            "path",
            p,
            [f"Unzip {p}.", f"Extract {p}.", f"Unpack the archive {p}.", f"Decompress {p}."],
        )

    def env_get(self) -> Sample:
        nm = self.rng.choice(self.envvars)
        return self._string_tool(
            "env_get",
            "code",
            "env_get",
            "name",
            nm,
            [
                f"Get the env variable '{nm}'.",
                f"Read the environment variable '{nm}'.",
                f"What is '{nm}' set to?",
                f"Show the value of '{nm}'.",
            ],
        )

    def make_dir(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool(
            "make_dir",
            "code",
            "make_dir",
            "path",
            p,
            [
                f"Create the directory {p}.",
                f"Make a directory {p}.",
                f"mkdir {p}.",
                f"Create folder {p}.",
            ],
        )

    def list_processes(self) -> Sample:
        s = self._tool_sample("list_processes", "code", "list_processes", {})
        s.prompt = self._split_choice(
            [
                "List the running processes.",
                "Show running processes.",
                "What processes are running?",
                "List all processes.",
                "Show me the process list.",
            ],
            [
                "Could you list the processes?",
                "Show me running processes.",
                "Which processes are active?",
                "Display all processes.",
                "Give me the process list.",
            ],
        )
        return s

    def docker_run(self) -> Sample:
        img = self.rng.choice(self.images)
        return self._string_tool(
            "docker_run",
            "code",
            "docker_run",
            "image",
            img,
            [
                f"Run a Docker container from '{img}'.",
                f"Run the '{img}' image.",
                f"Start a container from '{img}'.",
                f"docker run '{img}'.",
            ],
        )

    # --- parallel / two-tool calls ("do X and Y" — what people actually want) ---
    _PARALLEL_POOL = (
        "weather",
        "web_search",
        "web_search_implicit",
        "define",
        "play_music",
        "get_news",
        "read_file",
        "run_tests",
        "set_reminder",
        "set_timer",
        "calendar_event",
        "send_email",
        "open_url",
        "notion_write",
        "slack_send",
        "jira_issue",
        "grep_search",
        "git_commit",
        "calc",
        "run_command",
        # computer-use + modern tools (single groundable clause each)
        "screenshot",
        "click",
        "type_text",
        "key_press",
        "scroll",
        "wait",
        "open_app",
        "move_cursor",
        "double_click",
        "run_python",
        "edit_file",
        "git_diff",
        "git_status",
        "install_package",
        "find_files",
        "list_dir",
        "kill_process",
        "sql_query",
        "env_get",
        "make_dir",
        "unzip",
        "docker_run",
    )

    def parallel(self) -> Sample:
        """One user turn that needs TWO tool calls, joined by 'and'. Each clause is a standalone
        single-tool request so it splits/grounds cleanly (no value contains ' and ')."""
        for _ in range(20):
            a = getattr(self, self.rng.choice(self._PARALLEL_POOL))()
            b = getattr(self, self.rng.choice(self._PARALLEL_POOL))()
            p2 = b.prompt
            p2 = (p2[0].lower() + p2[1:]) if p2 else p2
            prompt = a.prompt.rstrip(".?! ") + " and " + p2
            if prompt.count(" and ") != 1:  # a value sneaked in an 'and' — retry
                continue
            calls = [
                {"name": a.ref_name, "arguments": json.loads(a.ref_args or "{}")},
                {"name": b.ref_name, "arguments": json.loads(b.ref_args or "{}")},
            ]
            target = " ".join(_tool_target(c["name"], c["arguments"]) for c in calls)
            return Sample(
                "parallel", "parallel", prompt, "tool", target, a.ref_name, a.ref_args, calls
            )
        return a  # fallback (rare)

    def text(self) -> Sample:
        name = self.rng.choice(self.names)
        choice = self.rng.choice(["hello", "morning", "name"])
        if choice == "hello":
            return Sample("text", "text", f"Say hello to {name}.", "text", f"Hello, {name}!")
        if choice == "morning":
            return Sample(
                "text", "text", f"Greet {name} in the morning.", "text", f"Good morning, {name}!"
            )
        prompt = "What is your name?" if self.split == "train" else "Could you introduce yourself?"
        return Sample("text", "text", prompt, "text", "I am LocalAgent.")

    def no_tool(self) -> Sample:
        """Diverse restraint examples where calling any tool is wrong."""

        name = self.rng.choice(self.names)
        topic = self.rng.choice(self.topics)
        if self.mode == PAPER_TRAIN_V2_MODE:
            amount = self.rng.choice(PAPER_V2_SCROLL_AMOUNTS_TRAIN)
            cue, _smooth = self.rng.choice(PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN)
            prompt, target = self.rng.choice(
                [
                    (
                        f"{name} already handled {topic}; do not scroll {amount} screen lengths "
                        f"or {cue}. Just acknowledge.",
                        "Acknowledged.",
                    ),
                    (
                        f"Do not execute anything for {topic}, including a {amount}-screen-length "
                        f"scroll or '{cue}'. Reply OK to {name}.",
                        "OK",
                    ),
                    (
                        f"The {topic} workflow is complete. Ignore the old instruction to scroll "
                        f"{amount} screen lengths and {cue}; confirm for {name}.",
                        "Confirmed.",
                    ),
                    (
                        f"Never mind the {amount}-screen-length scroll for {topic}. Do not {cue}; "
                        f"{name} only needs a text acknowledgement.",
                        "Understood.",
                    ),
                    (
                        f"For {name}: the note about {topic}, scrolling {amount} screen lengths, "
                        f"and '{cue}' is reference text only. Take no action.",
                        "Acknowledged.",
                    ),
                    (
                        f"Please state that no action is needed for {topic}. In particular, do not "
                        f"scroll {amount} screen lengths or {cue}. Address {name}.",
                        f"No action is needed, {name}.",
                    ),
                    (
                        f"{name} cancelled the {topic} task before the planned {amount}-screen-"
                        f"length scroll. Do not {cue}; respond with 'Cancelled.'",
                        "Cancelled.",
                    ),
                    (
                        f"Treat 'scroll {amount} screen lengths and {cue}' as quoted background "
                        f"about {topic}, not a command. Tell {name} you understand.",
                        f"Understood, {name}.",
                    ),
                ]
            )
            return Sample("no_tool", "text", prompt, "text", target)
        prefix = self._split_choice(
            ["", "For now, ", "A quick note: "],
            ["At this point, ", "For this request, ", "One final note: "],
        )
        body, target = self.rng.choice(
            [
                (f"Thanks for helping with {topic}, {name}.", "You're welcome!"),
                (
                    f"No action is needed. Just acknowledge this note about {topic}.",
                    "Acknowledged.",
                ),
                (f"Do not use a tool; reply OK to {name}.", "OK"),
                (f"{name} says the {topic} task is already complete. Confirm that.", "Confirmed."),
                (f"Never mind about {topic}; I handled it myself.", "Understood."),
            ]
        )
        prompt = prefix + body
        return Sample("no_tool", "text", prompt, "text", target)

    # ---- dataset assembly ----
    def makers(self):
        m = [
            self.weather,
            self.calc,
            self.web_search,
            self.web_search_implicit,
            self.planner,
            self.define,
            self.play_music,
            self.get_news,
            self.read_file,
            self.write_file,
            self.grep_search,
            self.run_command,
            self.git_commit,
            self.run_tests,
            self.set_reminder,
            self.set_timer,
            self.calendar_event,
            self.send_email,
            self.open_url,
            self.notion_write,
            self.slack_send,
            self.jira_issue,
            # computer-use family
            self.screenshot,
            self.click,
            self.double_click,
            self.type_text,
            self.key_press,
            self.scroll,
            self.drag,
            self.wait,
            self.move_cursor,
            self.open_app,
            # modern dev / agentic tools
            self.run_python,
            self.edit_file,
            self.apply_patch,
            self.http_request,
            self.sql_query,
            self.list_dir,
            self.find_files,
            self.git_diff,
            self.git_status,
            self.install_package,
            self.kill_process,
            self.read_clipboard,
            self.write_clipboard,
            self.download_file,
            self.unzip,
            self.env_get,
            self.make_dir,
            self.list_processes,
            self.docker_run,
            self.parallel,
            self.text,
        ]
        if self.level >= 2:
            m.append(self.no_tool)
        if self.mode == PAPER_TRAIN_V2_MODE:
            m.append(self.precise_scroll_v2)
        return m

    # --- multi-turn trajectory helpers ---------------------------------------------------
    @staticmethod
    def _A(name, args):
        return Message(role=Role.assistant, tool_calls=[ToolCall(name, args)])

    @staticmethod
    def _T(resp):
        return Message(role=Role.tool, tool_response=resp)

    def _U(self, content):
        if self.split == "eval":
            content = self.rng.choice(self._EVAL_EPISODE_WRAPPERS).format(request=content)
        return Message(role=Role.user, content=content)

    @staticmethod
    def _F(content):  # final / intermediate assistant text turn
        return Message(role=Role.assistant, content=content)

    # Episode-type registries. Each builder returns a list[Message]. Builders whose name is in
    # `_PTR_*` ground at least one follow-up argument from a TOOL RESPONSE (pointer-head case).
    _CODING_TYPES = (
        "debug",
        "grep_read",
        "test_commit",
        "implement_feature",
        "fix_failing_test",
        "refactor_rename",
        "review_pr",
        "dependency_bump",
        "trace_import",
        "lint_and_fix",
    )
    _PRODUCTIVITY_TYPES = (
        "research_summarize",
        "schedule_and_notify",
        "triage_issue",
        "news_brief",
        "open_and_note",
        "remind_and_slack",
    )
    # All plan-episode types (length-bucketed). Backwards-compatible: the original three
    # (plan_read_test_commit, plan_research, plan_fix_test) are a subset. The gold "plan" of any
    # episode is recoverable via ``episode_plan`` (ordered tool names). Keep this tuple in sync
    # with ``_PLAN_BUILDERS`` below — ``test_planner_data`` asserts they match exactly.
    _PLANNER_TYPES = (
        "plan_no_tool_thanks",
        "plan_no_tool_greet",
        "plan_single_read",
        "plan_single_search",
        "plan_single_timer",
        "plan_research_record",
        "plan_research_share",
        "plan_schedule_notify",
        "plan_remind_timer",
        "plan_read_run",
        "plan_read_test_commit",
        "plan_research",
        "plan_locate_inspect_edit",
        "plan_search_open_note",
        "plan_gui_open_click",
        "plan_fix_test",
        "plan_implement",
        "plan_gui_login",
    )

    # ---- coding episode builders (return list[Message]) ----
    def _ep_debug(self):
        path = self.rng.choice(self.paths)
        return [
            self._U(f"There's a bug in {path}, investigate it."),
            self._A("read_file", {"path": path}),
            self._T("def handler():\n    return None  # suspicious"),
            self._A("run_tests", {}),
            self._T("FAILED tests/test_handler.py::test_ok"),
            self._F(f"The bug is in {path}: handler() returns None."),
        ]

    def _ep_grep_read(self):  # follow-up path grounded from the tool response (pointer case)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Where is '{pat}' used? Open that file."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:12:    {pat} appears here"),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._F(f"'{pat}' is used in {path}."),
        ]

    def _ep_test_commit(self):
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Run the tests and commit with message '{msg}'."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed abc123."),
            self._F(f"Done — committed '{msg}'."),
        ]

    def _ep_implement_feature(self):  # grep -> read -> write -> test -> commit (pointer case)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Implement '{msg}'; the relevant code uses '{pat}'."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:20:    {pat} ..."),
            self._A("read_file", {"path": path}),
            self._T("<current implementation>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed def456."),
            self._F(f"Implemented and committed '{msg}'."),
        ]

    def _ep_fix_failing_test(self):  # run_tests -> read -> write -> run_tests
        path = self.rng.choice(self.paths)
        return [
            self._U(f"A test is failing in {path}, please fix it."),
            self._A("run_tests", {}),
            self._T(f"FAILED {path}::test_case"),
            self._A("read_file", {"path": path}),
            self._T("def f(): return None"),
            self._A("write_file", {"path": path}),
            self._T("patched."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._F(f"Fixed the failing test in {path}."),
        ]

    def _ep_refactor_rename(
        self,
    ):  # grep -> (response has path) -> read -> write -> tests (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Rename the symbol '{pat}'; find where it lives and update it."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:34:    {pat} = ...  # definition"),
            self._A("read_file", {"path": path}),
            self._T("<current source>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._F(f"Renamed '{pat}' in {path} and tests pass."),
        ]

    def _ep_review_pr(self):  # read -> (response = diff) -> comment text turn
        path = self.rng.choice(self.paths)
        return [
            self._U(f"Review the changes in {path}."),
            self._A("read_file", {"path": path}),
            self._T(f"@@ -1,3 +1,4 @@\n+    log.debug('x')\n     return value  # in {path}"),
            self._F(f"Looks good: {path} adds a debug log; no blocking issues."),
        ]

    def _ep_dependency_bump(self):  # read config -> write config -> install -> tests -> commit
        path = self.rng.choice(self.paths)
        cmd = self.rng.choice(self.commands)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Bump a dependency in {path}, install, test, then commit '{msg}'."),
            self._A("read_file", {"path": path}),
            self._T("requests==2.0.0"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("run_command", {"command": cmd}),
            self._T("install complete."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed 9f8e7d."),
            self._F(f"Bumped the dependency and committed '{msg}'."),
        ]

    def _ep_trace_import(self):  # grep -> (response has path) -> read -> text (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Trace where '{pat}' is imported and open that module."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:1:from x import {pat}"),
            self._A("read_file", {"path": path}),
            self._T("<module source>"),
            self._F(f"'{pat}' is imported in {path}."),
        ]

    def _ep_lint_and_fix(
        self,
    ):  # run_command(lint) -> (response has path) -> read -> write (pointer)
        path = self.rng.choice(self.paths)
        cmd = self.rng.choice(self.commands)
        return [
            self._U(f"Run '{cmd}' and fix whatever file it flags."),
            self._A("run_command", {"command": cmd}),
            self._T(f"{path}:7: E501 line too long"),
            self._A("read_file", {"path": path}),
            self._T("<offending line>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._F(f"Fixed the lint issue in {path}."),
        ]

    # ---- productivity / computer-use episode builders ----
    def _ep_research_summarize(
        self,
    ):  # web_search -> (response has url) -> open_url -> text (pointer)
        query = self.rng.choice(self.queries)
        url = self.rng.choice(self.urls)
        return [
            self._U(f"Research {query} and open the top result."),
            self._A("web_search", {"query": query}),
            self._T(f"1. {url} — a relevant overview of {query}"),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._F(f"Top result for {query} is {url}."),
        ]

    def _ep_schedule_and_notify(self):  # calendar_event -> send_email -> text
        title = self.rng.choice(self.events)
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Schedule '{title}' and email {nm} about it."),
            self._A("calendar_event", {"title": title}),
            self._T("event created."),
            self._A("send_email", {"recipient": nm}),
            self._T("email sent."),
            self._F(f"Scheduled '{title}' and emailed {nm}."),
        ]

    def _ep_triage_issue(self):  # jira_issue -> slack_send -> text
        summary = self.rng.choice(self.jira)
        msg = self.rng.choice(self.slack)
        return [
            self._U(f"File a Jira issue for '{summary}' and ping the team on Slack '{msg}'."),
            self._A("jira_issue", {"summary": summary}),
            self._T("created PROJ-123."),
            self._A("slack_send", {"message": msg}),
            self._T("message posted."),
            self._F(f"Filed '{summary}' and notified the team."),
        ]

    def _ep_news_brief(self):  # get_news -> (response has url) -> open_url -> text (pointer)
        topic = self.rng.choice(self.topics)
        url = self.rng.choice(self.urls)
        return [
            self._U(f"Get the latest news on {topic} and open the top story."),
            self._A("get_news", {"topic": topic}),
            self._T(f"Top story: {url} covers {topic}"),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._F(f"Top {topic} story is at {url}."),
        ]

    def _ep_open_and_note(self):  # open_url -> notion_write -> text
        url = self.rng.choice(self.urls)
        content = self.rng.choice(self.notion)
        return [
            self._U(f"Open {url} and note '{content}' in Notion."),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._A("notion_write", {"content": content}),
            self._T("note saved."),
            self._F(f"Opened {url} and saved '{content}' to Notion."),
        ]

    def _ep_remind_and_slack(self):  # set_reminder -> slack_send -> text
        task = self.rng.choice(self.tasks)
        msg = self.rng.choice(self.slack)
        return [
            self._U(f"Remind me to {task} and tell the team on Slack '{msg}'."),
            self._A("set_reminder", {"task": task}),
            self._T("reminder set."),
            self._A("slack_send", {"message": msg}),
            self._T("message posted."),
            self._F(f"Reminder set to {task} and team notified."),
        ]

    # ---- computer-use (GUI) episode builders ----
    # Realistic desktop/browser flows the planner can learn. Every arg is a literal substring of an
    # earlier turn (the user request or a tool response). Screenshots/keypresses interleave sensibly.
    def _ep_cu_login(self):  # screenshot -> click(field) -> type_text -> key_press(Enter)
        field = self.rng.choice([t for t in self.ui_targets if "field" in t] or self.ui_targets)
        text = self.rng.choice(self.typed_text)
        return [
            self._U(f"Log in: click '{field}', type '{text}', then press Enter."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._A("click", {"target": field}),
            self._T("focused."),
            self._A("type_text", {"text": text}),
            self._T("typed."),
            self._A("key_press", {"key": "Enter"}),
            self._T("submitted."),
            self._F(f"Logged in: typed '{text}' into '{field}'."),
        ]

    def _ep_cu_open_and_click(self):  # open_app -> click -> screenshot
        app = self.rng.choice(self.apps)
        target = self.rng.choice(self.ui_targets)
        return [
            self._U(f"Open '{app}', click '{target}', and take a screenshot."),
            self._A("open_app", {"name": app}),
            self._T("app launched."),
            self._A("click", {"target": target}),
            self._T("clicked."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._F(f"Opened '{app}' and clicked '{target}'."),
        ]

    def _ep_cu_search_box(self):  # click(search box) -> type_text -> key_press(Enter) -> screenshot
        text = self.rng.choice(self.typed_text)
        return [
            self._U(f"Click 'the Search box', search for '{text}', and capture the results."),
            self._A("click", {"target": "the Search box"}),
            self._T("focused."),
            self._A("type_text", {"text": text}),
            self._T("typed."),
            self._A("key_press", {"key": "Enter"}),
            self._T("searching."),
            self._A("screenshot", {}),
            self._T("<results captured>"),
            self._F(f"Searched for '{text}'."),
        ]

    def _ep_cu_scroll_click(self):  # scroll(down) -> screenshot -> click
        target = self.rng.choice(self.ui_targets)
        return [
            self._U(f"Scroll down, take a screenshot, then click '{target}'."),
            self._A("scroll", {"direction": "down"}),
            self._T("scrolled."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._A("click", {"target": target}),
            self._T("clicked."),
            self._F(f"Scrolled down and clicked '{target}'."),
        ]

    def _ep_cu_drag_drop(self):  # screenshot -> drag(source, dest)
        src, dst = self.rng.sample(self.ui_targets, 2)
        return [
            self._U(f"Take a screenshot then drag '{src}' to '{dst}'."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._A("drag", {"source": src, "dest": dst}),
            self._T("dropped."),
            self._F(f"Dragged '{src}' onto '{dst}'."),
        ]

    def _computer_use_builders(self):
        return {
            "cu_login": self._ep_cu_login,
            "cu_open_and_click": self._ep_cu_open_and_click,
            "cu_search_box": self._ep_cu_search_box,
            "cu_scroll_click": self._ep_cu_scroll_click,
            "cu_drag_drop": self._ep_cu_drag_drop,
        }

    _COMPUTER_USE_TYPES = (
        "cu_login",
        "cu_open_and_click",
        "cu_search_box",
        "cu_scroll_click",
        "cu_drag_drop",
    )

    def computer_use_episode(self) -> Conversation:
        """A short multi-turn GUI trajectory (open app / click / type / key / scroll / drag /
        screenshot). All args are text-grounded substrings; no pixel coordinates."""
        which = self.rng.choice(self._COMPUTER_USE_TYPES)
        msgs = self._computer_use_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "computer_use_episode", "type": which})

    def computer_use_episodes(self, n: int) -> list[Conversation]:
        return [self.computer_use_episode() for _ in range(n)]

    # ---- planner-then-execute episode builders ----
    # The plan text is deterministic/canonical so it is exactly learnable and scorable.
    @staticmethod
    def _plan(steps):
        return "Plan: " + " ".join(f"{i + 1}) {s}" for i, s in enumerate(steps))

    # ---- richer multi-step PLAN episodes (planner -> action decomposition, stage 1) ----
    # Each builder returns list[Message]. The gold "plan" is the ordered projection of the
    # episode's tool-call turns onto tool NAMES (see ``episode_plan`` below); each step's args
    # are groundable — copyable either from the composite user request or from an EARLIER tool
    # response (the pointer case, where a path/url is "returned" then consumed downstream).
    # Plan lengths span 1..4 so the planner learns plan-LENGTH control, not just a fixed shape.

    # --- 2-step plans ---
    def _ep_plan_research_record(self):  # web_search -> notion_write (research then record)
        query = self.rng.choice(self.queries)
        content = self.rng.choice(self.notion)
        return [
            self._U(f"Research {query} then note '{content}' in Notion."),
            self._F(self._plan(["search the web", "save a Notion note"])),
            self._A("web_search", {"query": query}),
            self._T(f"Results for {query}."),
            self._A("notion_write", {"content": content}),
            self._T("note saved."),
            self._F(f"Researched {query} and noted '{content}'."),
        ]

    def _ep_plan_research_share(self):  # web_search -> send_email (research then share)
        query = self.rng.choice(self.queries)
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Look up {query} and email {nm} about it."),
            self._F(self._plan(["search the web", "email the result"])),
            self._A("web_search", {"query": query}),
            self._T(f"Results for {query}."),
            self._A("send_email", {"recipient": nm}),
            self._T("email sent."),
            self._F(f"Looked up {query} and emailed {nm}."),
        ]

    def _ep_plan_schedule_notify(self):  # calendar_event -> send_email (schedule then notify)
        title = self.rng.choice(self.events)
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Schedule '{title}' then notify {nm} by email."),
            self._F(self._plan(["create a calendar event", "send an email"])),
            self._A("calendar_event", {"title": title}),
            self._T("event created."),
            self._A("send_email", {"recipient": nm}),
            self._T("email sent."),
            self._F(f"Scheduled '{title}' and notified {nm}."),
        ]

    def _ep_plan_remind_timer(self):  # set_reminder -> set_timer
        task = self.rng.choice(self.tasks)
        dur = self.rng.choice(self.durations)
        return [
            self._U(f"Remind me to {task} and set a timer for {dur}."),
            self._F(self._plan(["set a reminder", "set a timer"])),
            self._A("set_reminder", {"task": task}),
            self._T("reminder set."),
            self._A("set_timer", {"duration": dur}),
            self._T("timer started."),
            self._F(f"Reminder to {task} set and timer for {dur} started."),
        ]

    def _ep_plan_read_run(self):  # read_file -> run_command
        path = self.rng.choice(self.paths)
        cmd = self.rng.choice(self.commands)
        return [
            self._U(f"Read {path} then run '{cmd}'."),
            self._F(self._plan(["read the file", "run the command"])),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._A("run_command", {"command": cmd}),
            self._T("command finished."),
            self._F(f"Read {path} and ran '{cmd}'."),
        ]

    # --- 3-step plans ---
    def _ep_plan_locate_inspect_edit(self):  # grep_search -> read_file -> write_file (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Find '{pat}', inspect that file, then edit it."),
            self._F(self._plan(["search the code", "read the file", "edit the file"])),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:9:    {pat} appears here"),  # path "returned" by grep
            self._A("read_file", {"path": path}),  # grounded from tool response
            self._T("<current source>"),
            self._A("write_file", {"path": path}),  # grounded from tool response
            self._T("written."),
            self._F(f"Located '{pat}' in {path} and edited it."),
        ]

    def _ep_plan_search_open_note(self):  # web_search -> open_url -> notion_write (pointer)
        query = self.rng.choice(self.queries)
        url = self.rng.choice(self.urls)
        content = self.rng.choice(self.notion)
        return [
            self._U(f"Research {query}, open the top link, then note '{content}'."),
            self._F(self._plan(["search the web", "open the top result", "save a Notion note"])),
            self._A("web_search", {"query": query}),
            self._T(f"1. {url} — overview of {query}"),  # url "returned" by search
            self._A("open_url", {"url": url}),  # grounded from tool response
            self._T("<page loaded>"),
            self._A("notion_write", {"content": content}),
            self._T("note saved."),
            self._F(f"Researched {query}, opened {url}, noted '{content}'."),
        ]

    # --- 4-step plans ---
    def _ep_plan_read_test_commit(self):  # 3-step, existing one (kept, lightly varied phrasing)
        path = self.rng.choice(self.paths)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Read {path}, run the tests, then commit '{msg}'."),
            self._F(self._plan(["read the file", "run tests", "commit"])),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed 1a2b3c."),
            self._F(f"Read {path}, tests pass, committed '{msg}'."),
        ]

    def _ep_plan_research(self):  # 3-step plan -> search -> (url) -> open_url -> summary (pointer)
        query = self.rng.choice(self.queries)
        url = self.rng.choice(self.urls)
        return [
            self._U(f"Look into {query}: search, open the best link, then summarize."),
            self._F(self._plan(["search the web", "open the top result", "summarize"])),
            self._A("web_search", {"query": query}),
            self._T(f"1. {url} — the best overview of {query}"),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._F(f"Summary: {url} is the best source on {query}."),
        ]

    def _ep_plan_fix_test(self):  # 4-step: run_tests -> read -> write -> run_tests
        path = self.rng.choice(self.paths)
        return [
            self._U(f"A test in {path} is broken — make a plan and fix it."),
            self._F(self._plan(["run tests", "read the file", "fix it", "run tests again"])),
            self._A("run_tests", {}),
            self._T(f"FAILED {path}::test_case"),
            self._A("read_file", {"path": path}),
            self._T("def f(): return None"),
            self._A("write_file", {"path": path}),
            self._T("patched."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._F(f"Fixed the failing test in {path}."),
        ]

    def _ep_plan_implement(self):  # 4-step: grep -> (path) -> read -> write -> git_commit (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Implement '{msg}' where '{pat}' lives, then commit."),
            self._F(self._plan(["search the code", "read the file", "edit the file", "commit"])),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:20:    {pat} ..."),  # path "returned" by grep
            self._A("read_file", {"path": path}),
            self._T("<current implementation>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed def456."),
            self._F(f"Implemented and committed '{msg}'."),
        ]

    # --- GUI plan episodes (computer-use planner flows) ---
    def _ep_plan_gui_login(self):  # 4-step: click -> type_text -> key_press -> screenshot
        field = self.rng.choice([t for t in self.ui_targets if "field" in t] or self.ui_targets)
        text = self.rng.choice(self.typed_text)
        return [
            self._U(f"Log in: click '{field}', type '{text}', press Enter, then screenshot."),
            self._F(
                self._plan(["click the field", "type the text", "press Enter", "take a screenshot"])
            ),
            self._A("click", {"target": field}),
            self._T("focused."),
            self._A("type_text", {"text": text}),
            self._T("typed."),
            self._A("key_press", {"key": "Enter"}),
            self._T("submitted."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._F("Logged in and captured the screen."),
        ]

    def _ep_plan_gui_open_click(self):  # 3-step: open_app -> click -> screenshot
        app = self.rng.choice(self.apps)
        target = self.rng.choice(self.ui_targets)
        return [
            self._U(f"Open '{app}', click '{target}', then take a screenshot."),
            self._F(self._plan(["open the app", "click the element", "take a screenshot"])),
            self._A("open_app", {"name": app}),
            self._T("app launched."),
            self._A("click", {"target": target}),
            self._T("clicked."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._F(f"Opened '{app}' and clicked '{target}'."),
        ]

    # --- 1-step plans (teach the planner to STOP after one tool) ---
    def _ep_plan_single_read(self):
        path = self.rng.choice(self.paths)
        return [
            self._U(f"Just read {path}."),
            self._F(self._plan(["read the file"])),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._F(f"Read {path}."),
        ]

    def _ep_plan_single_search(self):
        query = self.rng.choice(self.queries)
        return [
            self._U(f"Only search the web for {query}."),
            self._F(self._plan(["search the web"])),
            self._A("web_search", {"query": query}),
            self._T(f"Results for {query}."),
            self._F(f"Searched for {query}."),
        ]

    def _ep_plan_single_timer(self):
        dur = self.rng.choice(self.durations)
        return [
            self._U(f"Set a timer for {dur}, nothing else."),
            self._F(self._plan(["set a timer"])),
            self._A("set_timer", {"duration": dur}),
            self._T("timer started."),
            self._F(f"Timer for {dur} started."),
        ]

    # --- 0-step "plans": a trivial request answered with text, no tool (don't over-plan) ---
    def _ep_plan_no_tool_thanks(self):
        return [
            self._U("Thanks for your help!"),
            self._F("Plan: just reply."),
            self._F("You're welcome!"),
        ]

    def _ep_plan_no_tool_greet(self):
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Just say hello to {nm}."),
            self._F("Plan: just reply."),
            self._F(f"Hello, {nm}!"),
        ]

    # length-bucketed registry; ``plan_episode`` samples across buckets for plan-length variety
    _PLAN_BUILDERS: ClassVar[dict[str, tuple[str, int]]] = {
        # 0-step (text only, no tool)
        "plan_no_tool_thanks": ("_ep_plan_no_tool_thanks", 0),
        "plan_no_tool_greet": ("_ep_plan_no_tool_greet", 0),
        # 1-step
        "plan_single_read": ("_ep_plan_single_read", 1),
        "plan_single_search": ("_ep_plan_single_search", 1),
        "plan_single_timer": ("_ep_plan_single_timer", 1),
        # 2-step
        "plan_research_record": ("_ep_plan_research_record", 2),
        "plan_research_share": ("_ep_plan_research_share", 2),
        "plan_schedule_notify": ("_ep_plan_schedule_notify", 2),
        "plan_remind_timer": ("_ep_plan_remind_timer", 2),
        "plan_read_run": ("_ep_plan_read_run", 2),
        # 3-step
        "plan_read_test_commit": ("_ep_plan_read_test_commit", 3),
        "plan_research": ("_ep_plan_research", 3),
        "plan_locate_inspect_edit": ("_ep_plan_locate_inspect_edit", 3),
        "plan_search_open_note": ("_ep_plan_search_open_note", 3),
        "plan_gui_open_click": ("_ep_plan_gui_open_click", 3),
        # 4-step
        "plan_fix_test": ("_ep_plan_fix_test", 4),
        "plan_implement": ("_ep_plan_implement", 4),
        "plan_gui_login": ("_ep_plan_gui_login", 4),
    }

    def _coding_builders(self):
        return {
            "debug": self._ep_debug,
            "grep_read": self._ep_grep_read,
            "test_commit": self._ep_test_commit,
            "implement_feature": self._ep_implement_feature,
            "fix_failing_test": self._ep_fix_failing_test,
            "refactor_rename": self._ep_refactor_rename,
            "review_pr": self._ep_review_pr,
            "dependency_bump": self._ep_dependency_bump,
            "trace_import": self._ep_trace_import,
            "lint_and_fix": self._ep_lint_and_fix,
        }

    def _productivity_builders(self):
        return {
            "research_summarize": self._ep_research_summarize,
            "schedule_and_notify": self._ep_schedule_and_notify,
            "triage_issue": self._ep_triage_issue,
            "news_brief": self._ep_news_brief,
            "open_and_note": self._ep_open_and_note,
            "remind_and_slack": self._ep_remind_and_slack,
        }

    def _planner_builders(self):
        return {name: getattr(self, attr) for name, (attr, _len) in self._PLAN_BUILDERS.items()}

    def coding_episode(self) -> Conversation:
        """A short multi-turn coding tool-use trajectory: tool call -> tool response -> follow-up.
        Some follow-up args are grounded in the *tool response*, not the user turn (the case
        only a learned pointer head can handle)."""
        which = self.rng.choice(self._CODING_TYPES)
        msgs = self._coding_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "coding_episode", "type": which})

    def productivity_episode(self) -> Conversation:
        """A multi-turn computer-use / productivity trajectory (calendar, email, browser, Slack,
        Notion, Jira). Several ground a follow-up arg (a URL) from the tool response."""
        which = self.rng.choice(self._PRODUCTIVITY_TYPES)
        msgs = self._productivity_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "productivity_episode", "type": which})

    def _build_plan_episode(self, which: str) -> Conversation:
        """Build a named plan episode and tag meta with its gold plan (ordered tool names) and
        plan length, both *derived* from the episode's tool-call sequence (no schema change)."""
        msgs = self._planner_builders()[which]()
        plan = episode_plan(Conversation(messages=msgs))
        return Conversation(
            messages=msgs,
            meta={"kind": "planner_episode", "type": which, "plan": plan, "plan_len": len(plan)},
        )

    def planner_episode(self) -> Conversation:
        """A planner-then-execute trajectory: a canonical NUMBERED text plan turn, then each step
        as a tool-call turn with tool responses between, then a final summary (teaches plan->act).
        Backwards-compatible: only multi-step types (>=1 tool call, plan starts ``Plan: 1)``); the
        0-step "don't over-plan" cases are reached via ``plan_episode``/``plan_episodes`` instead.
        ``meta['plan']`` is the gold ordered tool-name list; ``meta['plan_len']`` its length."""
        multi = [t for t, (_a, ln) in self._PLAN_BUILDERS.items() if ln >= 1]
        which = self.rng.choice(multi)
        return self._build_plan_episode(which)

    def plan_episode(self) -> Conversation:
        """Sample a plan episode with *plan-length variety*: pick a length bucket (0..4) uniformly,
        then a type within it, so the planner sees STOP-after-one and don't-over-plan cases, not
        only long plans. Same Conversation shape as ``planner_episode`` (alias-friendly)."""
        by_len: dict[int, list[str]] = {}
        for name, (_attr, ln) in self._PLAN_BUILDERS.items():
            by_len.setdefault(ln, []).append(name)
        length = self.rng.choice(sorted(by_len))
        which = self.rng.choice(by_len[length])
        return self._build_plan_episode(which)

    def coding_episodes(self, n: int) -> list[Conversation]:
        return [self.coding_episode() for _ in range(n)]

    def productivity_episodes(self, n: int) -> list[Conversation]:
        return [self.productivity_episode() for _ in range(n)]

    def planner_episodes(self, n: int) -> list[Conversation]:
        return [self.planner_episode() for _ in range(n)]

    def plan_episodes(self, n: int) -> list[Conversation]:
        """`n` plan episodes sampled for plan-length variety. Entry point for stage-1 planner
        training/eval: each Conversation's gold plan is ``meta['plan']`` (== ``episode_plan(ep)``)
        and its per-step grounded ToolCalls are the assistant tool-call turns, in order."""
        return [self.plan_episode() for _ in range(n)]

    def paper_v2_schema_episode(self) -> Conversation:
        """Train-only trajectory that exercises the v2 number/boolean scroll schema."""

        if self.mode != PAPER_TRAIN_V2_MODE:
            raise ValueError("paper_v2_schema_episode requires paper_train_v2 mode")
        direction = self.rng.choice(self._PAPER_V2_SCROLL_DIRECTIONS)
        amount = self.rng.choice(PAPER_V2_SCROLL_AMOUNTS_TRAIN)
        cue, smooth = self.rng.choice(PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN)
        request = self.rng.choice(
            [
                f"Scroll {direction} {amount} screen lengths and {cue}, then take a screenshot.",
                f"{cue.capitalize()}; move {direction} by {amount} screen lengths, "
                "then capture it.",
                f"Use a {amount}-screen-length scroll {direction} with this setting: {cue}. "
                "Afterward, capture the screen.",
                f"First scroll {direction} exactly {amount} screen lengths and {cue}. "
                "Then show me a screenshot.",
            ]
        )
        return Conversation(
            messages=[
                self._U(request),
                self._A(
                    "scroll",
                    {"direction": direction, "amount": amount, "smooth": smooth},
                ),
                self._T("scrolled."),
                self._A("screenshot", {}),
                self._T("<screen captured>"),
                self._F(
                    f"Scrolled {direction} by {amount} screen lengths and captured the screen."
                ),
            ],
            meta={
                "kind": "paper_v2_schema_episode",
                "type": "precise_scroll_then_capture",
                "stratum": "number_boolean_multi_turn",
            },
        )

    def paper_v2_recovery_episode(self) -> Conversation:
        """Rule-audited scripted recovery trace; no tool result is actually executed."""

        if self.mode != PAPER_TRAIN_V2_MODE:
            raise ValueError("paper_v2_recovery_episode requires paper_train_v2 mode")
        path = self.rng.choice(self.paths)
        direction = self.rng.choice(self._PAPER_V2_SCROLL_DIRECTIONS)
        amount = self.rng.choice(PAPER_V2_SCROLL_AMOUNTS_TRAIN)
        cue, smooth = self.rng.choice(PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN)
        request = self.rng.choice(
            [
                f"The browser check in {path} is failing. Run it, scroll {direction} by {amount} "
                f"screen lengths and {cue} to inspect the hidden state, patch it, then retry.",
                f"Recover the failing UI test in {path}: test first, then {cue} while scrolling "
                f"{direction} {amount} screen lengths, capture the state, fix it, and rerun.",
                f"Diagnose {path}. If the test fails, move {direction} by {amount} screen lengths "
                f"and {cue}, inspect a screenshot, edit the fixture, and verify with another run.",
            ]
        )
        return Conversation(
            messages=[
                self._U(request),
                self._A("run_tests", {}),
                self._T(f"FAILED {path}::test_hidden_state"),
                self._A(
                    "scroll",
                    {"direction": direction, "amount": amount, "smooth": smooth},
                ),
                self._T("scrolled to the hidden state."),
                self._A("screenshot", {}),
                self._T("<stale hidden state captured>"),
                self._A("edit_file", {"path": path}),
                self._T("fixture patched."),
                self._A("run_tests", {}),
                self._T("All tests passed."),
                self._F(f"Recovered the browser check in {path}; the retry passed."),
            ],
            meta={
                "kind": "paper_v2_recovery_episode",
                "type": "inspect_patch_retry",
                "stratum": "scripted_failure_recovery",
            },
        )

    def episodes(self, n: int, mix: bool = True) -> list[Conversation]:
        """Sample `n` multi-turn episodes. With ``mix=True`` (default) the pool spans coding +
        productivity + planner trajectories so the flywheel/eval see the full diversity; with
        ``mix=False`` it returns coding-only episodes (the original behaviour)."""
        if not mix:
            return [self.coding_episode() for _ in range(n)]
        if self.mode == PAPER_TRAIN_V2_MODE:
            builders = [
                self.coding_episode,
                self.productivity_episode,
                self.planner_episode,
                self.computer_use_episode,
                self.paper_v2_schema_episode,
                self.paper_v2_recovery_episode,
            ]
            weights = [0.3, 0.2, 0.15, 0.1, 0.15, 0.1]
            return [self.rng.choices(builders, weights=weights, k=1)[0]() for _ in range(n)]
        builders = [
            self.coding_episode,
            self.productivity_episode,
            self.planner_episode,
            self.computer_use_episode,
        ]
        # weight coding a bit higher (it has the most types), then productivity, planner, GUI
        weights = [0.4, 0.25, 0.2, 0.15]
        return [self.rng.choices(builders, weights=weights, k=1)[0]() for _ in range(n)]

    def generate(self, n: int) -> list[Sample]:
        makers = self.makers()
        out, seen = [], set()
        # round-robin categories by attempt index (NOT len(out)) so a low-diversity category
        # hitting duplicates can't freeze the selector; dedupe identical prompts.
        attempts = 0
        while len(out) < n and attempts < n * 80:
            s = makers[attempts % len(makers)]()
            attempts += 1
            if s.prompt in seen:
                continue
            seen.add(s.prompt)
            out.append(s)
        return out

    def generate_weighted(self, n: int, weights: dict) -> list[Sample]:
        """Sample categories with per-category weights (>1 oversamples). Drives failure-driven
        enrichment: the flywheel raises weights on the categories the model is failing."""
        makers = self.makers()
        cats = [mk().category for mk in makers]  # one call each to read its category
        w = [max(0.05, weights.get(c, 1.0)) for c in cats]
        out, seen, tries = [], set(), 0
        while len(out) < n and tries < n * 100:
            tries += 1
            s = self.rng.choices(makers, weights=w, k=1)[0]()
            if s.prompt in seen:
                continue
            seen.add(s.prompt)
            out.append(s)
        return out

    def generate_balanced(self, per_category: int) -> list[Sample]:
        """Up to `per_category` unique samples per category — for meaningful per-group eval."""
        from collections import Counter

        makers = self.makers()
        out, seen, cnt = [], set(), Counter()
        attempts, cap = 0, per_category * len(makers) * 300
        while len(out) < per_category * len(makers) and attempts < cap:
            s = makers[attempts % len(makers)]()
            attempts += 1
            if cnt[s.category] >= per_category or s.prompt in seen:
                continue
            seen.add(s.prompt)
            cnt[s.category] += 1
            out.append(s)
        return out


def _paper_train_v2_preflight(
    generator: Generator,
    n_irrelevant: int,
    minimum_conversations: dict[str, int],
    plan_length_minimums: dict[int, int],
) -> dict[str, object]:
    """Validate v2 stratum requirements against deterministic unique-template capacities."""

    mandatory_positive = (
        "boolean_arguments",
        "number_arguments",
        "paper_v2_schema_trajectories",
        "verified_error_recovery",
    )
    missing = [name for name in mandatory_positive if minimum_conversations.get(name, 0) < 1]
    if missing:
        raise ValueError(
            "paper_train_v2 requires positive coverage minima for "
            + ", ".join(sorted(missing))
        )
    if n_irrelevant < 1:
        raise ValueError("paper_train_v2 requires a positive irrelevance stratum")

    directions = len(Generator._PAPER_V2_SCROLL_DIRECTIONS)
    amounts = len({_slot_value_key(value) for value in PAPER_V2_SCROLL_AMOUNTS_TRAIN})
    cues = len({_slot_value_key(cue) for cue, _value in PAPER_V2_SCROLL_BOOLEAN_CUES_TRAIN})
    names = len({_slot_value_key(value) for value in generator.names})
    topics = len({_slot_value_key(value) for value in generator.topics})
    paths = len({_slot_value_key(value) for value in generator.paths})
    wait_values = len({_slot_value_key(value) for value in generator.wait_seconds})
    key_values = 10

    field_count = len([target for target in generator.ui_targets if "field" in target])
    if field_count == 0:
        field_count = len(generator.ui_targets)
    plan_capacities = {
        0: 1 + len(generator.names),
        1: len(generator.paths) + len(generator.queries) + len(generator.durations),
        2: (
            len(generator.queries) * len(generator.notion)
            + len(generator.queries) * len(generator.names)
            + len(generator.events) * len(generator.names)
            + len(generator.tasks) * len(generator.durations)
            + len(generator.paths) * len(generator.commands)
        ),
        3: (
            len(generator.paths) * len(generator.patterns)
            + len(generator.queries) * len(generator.urls) * len(generator.notion)
            + len(generator.paths) * len(generator.commits)
            + len(generator.queries) * len(generator.urls)
            + len(generator.apps) * len(generator.ui_targets)
        ),
        4: (
            len(generator.paths)
            + len(generator.paths) * len(generator.patterns) * len(generator.commits)
            + field_count * len(generator.typed_text)
        ),
    }
    capacities = {
        "enum_arguments": key_values * 5,
        "integer_arguments": wait_values * 5,
        "irrelevance_conversations": (
            names * topics * amounts * cues * Generator._PAPER_V2_NO_TOOL_TEMPLATE_COUNT
        ),
        "paper_v2_precise_scroll_single_turn": (
            directions * amounts * cues * Generator._PAPER_V2_SCROLL_TEMPLATE_COUNT
        ),
        "paper_v2_schema_trajectories": (
            directions * amounts * cues * Generator._PAPER_V2_SCHEMA_EPISODE_TEMPLATE_COUNT
        ),
        "verified_error_recovery": (
            paths
            * directions
            * amounts
            * cues
            * Generator._PAPER_V2_RECOVERY_EPISODE_TEMPLATE_COUNT
        ),
    }
    requirements = {
        "enum_arguments": minimum_conversations["enum_arguments"],
        "integer_arguments": minimum_conversations["integer_arguments"],
        "irrelevance_conversations": n_irrelevant,
        # Reservations are intentionally separate, even though each precise row covers both types.
        "paper_v2_precise_scroll_single_turn": (
            minimum_conversations["boolean_arguments"]
            + minimum_conversations["number_arguments"]
        ),
        "paper_v2_schema_trajectories": minimum_conversations[
            "paper_v2_schema_trajectories"
        ],
        "verified_error_recovery": minimum_conversations["verified_error_recovery"],
    }
    for label, required in requirements.items():
        if required > capacities[label]:
            raise ValueError(
                f"paper_train_v2 preflight capacity exceeded for {label}: "
                f"{required}/{capacities[label]}"
            )
    for length, required in plan_length_minimums.items():
        if required > plan_capacities[length]:
            raise ValueError(
                f"paper_train_v2 preflight capacity exceeded for plan length {length}: "
                f"{required}/{plan_capacities[length]}"
            )

    return {
        "status": "passed",
        "method": "closed-form unique prompt/trajectory template capacity bounds",
        "requirements": dict(sorted(requirements.items())),
        "capacities": dict(sorted(capacities.items())),
        "plan_length_requirements": {
            str(length): count for length, count in sorted(plan_length_minimums.items())
        },
        "plan_length_capacities": {
            str(length): count for length, count in sorted(plan_capacities.items())
        },
    }


# ---- plan helpers (free functions; no schema change — operate on the existing episode shape) --
def episode_plan(ep: Conversation) -> list[str]:
    """The gold PLAN of an episode: the ordered projection of its assistant tool-call turns onto
    tool NAMES. This is the stage-1 planner target — the planner emits this list, and the existing
    action decoder grounds each name into a concrete ToolCall. A 0-step plan (a trivial text-only
    request) yields ``[]``."""
    return [m.tool_calls[0].name for m in ep.messages if m.role == Role.assistant and m.tool_calls]


def episode_steps(ep: Conversation) -> list[ToolCall]:
    """The per-step grounded ToolCalls of an episode, in order — one per planned step. Zipped with
    ``episode_plan(ep)`` they give (tool_name, grounded_args) for each step."""
    return [m.tool_calls[0] for m in ep.messages if m.role == Role.assistant and m.tool_calls]


# ---- curriculum ordering (LFM2-style easy->hard) ---------------------------------------------
# LFM2 orders pretraining data by empirical success probability (easy first, hard later). We port
# the *principle* to tool-calling SFT with a transparent, deterministic proxy for difficulty built
# from signals already present in each Sample — no schema change, no model in the loop:
#
#   score = (W_PARALLEL  * (n_tool_calls - 1)     # >1 call (the "X and Y" parallel turns) is hard
#          + W_ARGS      * max(0, n_required_args - 1)  # multi-arg (e.g. weather city+unit) is hard
#          + W_HAS_ARG   * has_any_arg             # a single copy-arg call is harder than no-arg/text
#          + W_ABSTAIN   * is_abstention           # "don't call a tool" negatives are subtle
#          + W_PROMPT    * prompt_len_bucket)      # longer phrasings, mild tie-breaker
#
# A bare text turn or a no-arg tool call (run_tests) scores ~0 (easiest); a two-call parallel turn
# with copy args scores highest (hardest). Ties broken by a stable hash of the prompt so the order
# is fully deterministic and independent of the input list order.
CURRICULUM_WEIGHTS = {
    "parallel": 3.0,  # per extra tool call beyond the first
    "args": 1.5,  # per required arg beyond the first
    "has_arg": 0.6,  # single copy-arg call vs no-arg/text
    "abstain": 1.0,  # abstention / irrelevance negative
    "prompt": 0.25,  # per ~40-char bucket of prompt length (tie-breaker scale)
}


def difficulty_score(s: Sample, weights: dict | None = None) -> float:
    """Transparent easy->hard difficulty score for one SFT `Sample` (higher = harder).

    Uses only signals already on the Sample (number of tool calls, number/copy of args, abstention,
    prompt length). Deterministic and side-effect free. See ``CURRICULUM_WEIGHTS`` for the recipe.
    """
    w = weights or CURRICULUM_WEIGHTS
    # number of tool calls in the turn: parallel samples carry `calls`; single tool/text => 1.
    n_calls = len(s.calls) if s.calls else 1
    # required args across the call(s).
    if s.calls:
        n_args = sum(len(c.get("arguments", {})) for c in s.calls)
        has_arg = 1.0 if n_args > 0 else 0.0
    elif s.kind == "tool":
        try:
            args = json.loads(s.ref_args) if s.ref_args else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        n_args = len(args)
        has_arg = 1.0 if n_args > 0 else 0.0
    else:
        n_args, has_arg = 0, 0.0
    is_abstain = 1.0 if s.category == "no_tool" else 0.0
    prompt_bucket = len(s.prompt) / 40.0
    return (
        w["parallel"] * (n_calls - 1)
        + w["args"] * max(0, n_args - 1)
        + w["has_arg"] * has_arg
        + w["abstain"] * is_abstain
        + w["prompt"] * prompt_bucket
    )


def curriculum_order(samples: list, weights: dict | None = None) -> list:
    """Return `samples` reordered easy->hard by ``difficulty_score`` (ascending). Deterministic:
    ties are broken by a stable hash of (target, prompt), so the result does not depend on the
    input order. Opt-in — callers choose this over a shuffle. Does NOT mutate the input list."""
    import hashlib

    def _key(s):
        h = hashlib.sha1(f"{s.target}\x00{s.prompt}".encode()).hexdigest()
        return (difficulty_score(s, weights), h)

    return sorted(samples, key=_key)


def synthesize(config_path: str) -> None:  # CLI entry retained
    """Export deterministic, rule-audited canonical ``Conversation`` JSONL data."""

    import hashlib
    from collections import Counter
    from pathlib import Path

    import yaml

    from localagent.agent.schema_decode import validate as validate_arguments
    from localagent.agent.toolset import STANDARD_TOOLS
    from localagent.data.conversation_artifact import (
        CONVERSATION_SERIALIZATION,
        MANIFEST_KIND,
        MANIFEST_SCHEMA_VERSION,
        FileIdentity,
        self_hashed_manifest,
    )
    from localagent.data.pretrain_corpus import read_evaluation_denylist

    config_file = Path(config_path)
    config_payload = config_file.read_bytes()
    try:
        config_text = config_payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("synthetic-data config must be valid UTF-8") from exc
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise TypeError("synthetic-data config must be a mapping")
    config_identity = FileIdentity.from_bytes(config_payload)

    raw_holdouts = config.get("exact_prompt_holdouts", [])
    if not isinstance(raw_holdouts, list):
        raise TypeError("exact_prompt_holdouts must be a list")
    holdout_prompts: set[str] = set()
    holdout_artifacts: list[dict[str, object]] = []
    holdout_names: set[str] = set()
    holdout_paths: set[Path] = set()
    for index, entry in enumerate(raw_holdouts):
        if not isinstance(entry, dict):
            raise TypeError(f"exact_prompt_holdouts[{index}] must be a mapping")
        name = entry.get("name")
        declared_path = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if not isinstance(name, str) or not name:
            raise ValueError(f"exact_prompt_holdouts[{index}].name must be non-empty")
        if name in holdout_names:
            raise ValueError(f"duplicate exact prompt holdout name {name!r}")
        holdout_names.add(name)
        if not isinstance(declared_path, str) or not declared_path:
            raise ValueError(f"exact_prompt_holdouts[{index}].path must be non-empty")
        path = Path(declared_path)
        if not path.is_absolute():
            path = config_file.resolve().parent / path
        path = path.resolve()
        if path in holdout_paths:
            raise ValueError("the same exact prompt holdout file was configured more than once")
        holdout_paths.add(path)
        if not path.is_file():
            raise ValueError(f"exact prompt holdout artifact is missing: {path}")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ValueError(f"exact_prompt_holdouts[{index}].bytes is invalid")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"exact_prompt_holdouts[{index}].sha256 is invalid")
        actual_bytes = path.stat().st_size
        with path.open("rb") as handle:
            actual_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_bytes != expected_bytes:
            raise ValueError(f"exact prompt holdout {name!r} byte-size mismatch")
        if actual_sha256 != expected_sha256:
            raise ValueError(f"exact prompt holdout {name!r} SHA-256 mismatch")
        raw_prompts = read_evaluation_denylist(path)
        normalized_prompts = {_canonical_holdout_prompt(prompt) for prompt in raw_prompts}
        if not normalized_prompts or "" in normalized_prompts:
            raise ValueError(f"exact prompt holdout {name!r} contains no prompts")
        holdout_prompts.update(normalized_prompts)
        holdout_artifacts.append(
            {
                "name": name,
                "path": declared_path,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "normalized_entries": len(normalized_prompts),
            }
        )
    holdout_artifacts.sort(key=lambda artifact: str(artifact["name"]))
    holdout_prompts_sha256 = hashlib.sha256(
        json.dumps(
            sorted(holdout_prompts),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    generator_config = config.get("generator", {})
    if not isinstance(generator_config, dict):
        raise TypeError("generator must be a mapping")
    if generator_config.get("backend") not in {None, "deterministic_templates"}:
        raise NotImplementedError(
            "Only generator.backend=deterministic_templates is implemented; "
            "external teacher generation needs a separately audited adapter"
        )
    raw_mode = generator_config.get("mode")
    mode = Generator._LEGACY_MODE if raw_mode is None else raw_mode
    if not isinstance(mode, str):
        raise ValueError("generator.mode must be a string")
    mode_version = generator_config.get("mode_version")
    if mode == Generator._LEGACY_MODE:
        if mode_version is not None:
            raise ValueError("generator.mode_version is valid only for versioned generator modes")
    elif mode == PAPER_TRAIN_V2_MODE:
        if (
            isinstance(mode_version, bool)
            or not isinstance(mode_version, int)
            or mode_version != PAPER_TRAIN_V2_MODE_VERSION
        ):
            raise ValueError(
                "paper_train_v2 requires "
                f"generator.mode_version={PAPER_TRAIN_V2_MODE_VERSION}"
            )
    else:
        raise ValueError(f"unsupported deterministic generator mode: {mode!r}")
    verification = config.get("verification", {})
    if verification.get("model_based", False):
        raise NotImplementedError(
            "model_based verification requested but no verifier adapter is configured"
        )
    n_samples = int(config.get("n_samples", 5_000))
    seed = int(config.get("seed", 42))
    level = int(config.get("level", 5))
    split = str(config.get("split", "train"))
    if mode == PAPER_TRAIN_V2_MODE and split != "train":
        raise ValueError("paper_train_v2 is train-only; frozen evaluation rows must stay v1")
    slot_audit = _paper_train_v2_slot_audit() if mode == PAPER_TRAIN_V2_MODE else None
    multi_fraction = float(config.get("complexity", {}).get("multi_turn", 0.2))
    irrelevance_fraction = float(config.get("irrelevance_fraction", 0.15))
    coverage = config.get("coverage", {})
    minimum_conversations = coverage.get("minimum_conversations", {})
    allowed_minimums = {
        "parallel_calls",
        "integer_arguments",
        "enum_arguments",
        "tool_response_grounded_followups",
        "verified_error_recovery",
    }
    if mode == PAPER_TRAIN_V2_MODE:
        allowed_minimums.update(
            {
                "boolean_arguments",
                "number_arguments",
                "paper_v2_schema_trajectories",
            }
        )
    unknown_minimums = set(minimum_conversations) - allowed_minimums
    if unknown_minimums:
        raise ValueError(f"unknown coverage.minimum_conversations keys: {sorted(unknown_minimums)}")

    def non_negative_count(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    minimum_conversations = {
        name: non_negative_count(minimum_conversations.get(name, 0), f"coverage.{name}")
        for name in sorted(allowed_minimums)
    }
    raw_plan_minimums = coverage.get("plan_length_minimums", {})
    plan_length_minimums: dict[int, int] = {}
    supported_plan_lengths = {
        registered_length for _attribute, registered_length in Generator._PLAN_BUILDERS.values()
    }
    for raw_length, raw_count in raw_plan_minimums.items():
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid plan length bucket: {raw_length!r}") from exc
        if length not in supported_plan_lengths:
            raise ValueError(f"unsupported plan length bucket: {length}")
        plan_length_minimums[length] = non_negative_count(
            raw_count, f"coverage.plan_length_minimums.{length}"
        )
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if multi_fraction < 0 or irrelevance_fraction < 0:
        raise ValueError("dataset fractions must be non-negative")
    n_multi = round(n_samples * multi_fraction)
    n_irrelevant = round(n_samples * irrelevance_fraction)
    if n_multi + n_irrelevant > n_samples:
        raise ValueError("multi_turn + irrelevance fractions exceed 1")
    n_regular = n_samples - n_multi - n_irrelevant
    reserved_regular = (
        minimum_conversations["parallel_calls"]
        + minimum_conversations["integer_arguments"]
        + minimum_conversations["enum_arguments"]
    )
    if mode == PAPER_TRAIN_V2_MODE:
        reserved_regular += (
            minimum_conversations["boolean_arguments"]
            + minimum_conversations["number_arguments"]
        )
    reserved_multi = (
        minimum_conversations["tool_response_grounded_followups"]
        + minimum_conversations["verified_error_recovery"]
        + sum(plan_length_minimums.values())
    )
    if mode == PAPER_TRAIN_V2_MODE:
        reserved_multi += minimum_conversations["paper_v2_schema_trajectories"]
    if reserved_regular > n_regular:
        raise ValueError(
            f"single-turn coverage quotas need {reserved_regular} rows, only {n_regular} available"
        )
    if reserved_multi > n_multi:
        raise ValueError(
            f"multi-turn coverage quotas need {reserved_multi} rows, only {n_multi} available"
        )
    generator = Generator(level=level, seed=seed, split=split, mode=mode)
    tools = (
        build_paper_train_v2_tools(STANDARD_TOOLS)
        if mode == PAPER_TRAIN_V2_MODE
        else STANDARD_TOOLS
    )
    tool_map = {tool.name: tool for tool in tools}
    paper_v2_preflight = (
        _paper_train_v2_preflight(
            generator,
            n_irrelevant,
            minimum_conversations,
            plan_length_minimums,
        )
        if mode == PAPER_TRAIN_V2_MODE
        else None
    )

    def from_sample(sample: Sample) -> Conversation:
        if sample.kind == "tool":
            raw_calls = sample.calls or [
                {"name": sample.ref_name, "arguments": json.loads(sample.ref_args or "{}")}
            ]
            assistant = Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(name=call["name"], arguments=call["arguments"]) for call in raw_calls
                ],
            )
        else:
            assistant = Message(role=Role.assistant, content=sample.target)
        return Conversation(
            messages=[Message(role=Role.user, content=sample.prompt), assistant],
            tools=tools,
            meta={
                "category": sample.category,
                "group": sample.group,
                "kind": sample.kind,
                "split": split,
                "generator": "deterministic_templates",
                "difficulty": difficulty_score(sample),
            },
        )

    def valid(conv: Conversation) -> bool:
        for message in conv.messages:
            for call in message.tool_calls:
                spec = tool_map.get(call.name)
                if spec is None:
                    return False
                schema = spec.parameters
                if not validate_arguments(call.arguments, schema):
                    return False
        if conv.meta.get("kind") == "planner_episode":
            plan = episode_plan(conv)
            if conv.meta.get("plan") != plan or conv.meta.get("plan_len") != len(plan):
                return False
        return True

    regular: list[Sample] = []
    seen: set[str] = set()
    attempts = 0

    def prompt_is_held_out(prompt: str) -> bool:
        return _canonical_holdout_prompt(prompt) in holdout_prompts

    def reserve_samples(maker, count: int, label: str) -> None:
        nonlocal attempts
        target = len(regular) + count
        cap = attempts + max(1_000, count * 200)
        while len(regular) < target and attempts < cap:
            attempts += 1
            sample = maker()
            if sample.prompt in seen or prompt_is_held_out(sample.prompt):
                continue
            seen.add(sample.prompt)
            regular.append(sample)
        if len(regular) != target:
            raise RuntimeError(f"generator diversity exhausted for {label} quota")

    reserve_samples(
        generator.parallel,
        minimum_conversations["parallel_calls"],
        "parallel_calls",
    )
    reserve_samples(
        generator.wait,
        minimum_conversations["integer_arguments"],
        "integer_arguments",
    )
    reserve_samples(
        generator.key_press,
        minimum_conversations["enum_arguments"],
        "enum_arguments",
    )
    if mode == PAPER_TRAIN_V2_MODE:
        reserve_samples(
            generator.precise_scroll_v2,
            minimum_conversations["boolean_arguments"],
            "boolean_arguments",
        )
        reserve_samples(
            generator.precise_scroll_v2,
            minimum_conversations["number_arguments"],
            "number_arguments",
        )
    while len(regular) < n_regular and attempts < max(1_000, n_regular * 40):
        attempts += 1
        sample = generator.makers()[attempts % len(generator.makers())]()
        if (
            sample.category == "no_tool"
            or sample.prompt in seen
            or prompt_is_held_out(sample.prompt)
        ):
            continue
        seen.add(sample.prompt)
        regular.append(sample)
    irrelevant: list[Sample] = []
    while len(irrelevant) < n_irrelevant and attempts < max(5_000, n_samples * 80):
        attempts += 1
        sample = generator.no_tool()
        if sample.prompt in seen or prompt_is_held_out(sample.prompt):
            continue
        seen.add(sample.prompt)
        irrelevant.append(sample)
    if len(regular) != n_regular or len(irrelevant) != n_irrelevant:
        raise RuntimeError(
            f"generator diversity exhausted: regular={len(regular)}/{n_regular}, "
            f"irrelevance={len(irrelevant)}/{n_irrelevant}"
        )

    conversations = [from_sample(sample) for sample in [*regular, *irrelevant]]

    quota_episodes: list[Conversation] = []
    quota_episode_keys: set[str] = set()

    def contains_holdout_prompt(conversation: Conversation) -> bool:
        return any(
            message.role == Role.user
            and message.content is not None
            and prompt_is_held_out(message.content)
            for message in conversation.messages
        )

    def reserve_episodes(builder, count: int, label: str) -> None:
        target = len(quota_episodes) + count
        tries = 0
        while len(quota_episodes) < target and tries < max(1_000, count * 200):
            tries += 1
            episode = builder()
            key = episode.to_json()
            if key in quota_episode_keys or contains_holdout_prompt(episode):
                continue
            quota_episode_keys.add(key)
            quota_episodes.append(episode)
        if len(quota_episodes) != target:
            raise RuntimeError(f"generator diversity exhausted for {label} quota")

    response_grounded_types = (
        ("coding", "grep_read"),
        ("coding", "implement_feature"),
        ("coding", "refactor_rename"),
        ("coding", "trace_import"),
        ("productivity", "research_summarize"),
        ("productivity", "news_brief"),
        ("planner", "plan_locate_inspect_edit"),
        ("planner", "plan_search_open_note"),
        ("planner", "plan_research"),
        ("planner", "plan_implement"),
    )
    response_index = 0

    def response_grounded_episode() -> Conversation:
        nonlocal response_index
        kind, which = response_grounded_types[response_index % len(response_grounded_types)]
        response_index += 1
        if kind == "coding":
            return Conversation(
                messages=generator._coding_builders()[which](),
                meta={"kind": "coding_episode", "type": which},
            )
        if kind == "productivity":
            return Conversation(
                messages=generator._productivity_builders()[which](),
                meta={"kind": "productivity_episode", "type": which},
            )
        return generator._build_plan_episode(which)

    reserve_episodes(
        response_grounded_episode,
        minimum_conversations["tool_response_grounded_followups"],
        "tool_response_grounded_followups",
    )

    recovery_index = 0

    def verified_recovery_episode() -> Conversation:
        nonlocal recovery_index
        recovery_index += 1
        if mode == PAPER_TRAIN_V2_MODE:
            return generator.paper_v2_recovery_episode()
        if recovery_index % 2:
            which = "fix_failing_test"
            return Conversation(
                messages=generator._coding_builders()[which](),
                meta={"kind": "coding_episode", "type": which},
            )
        return generator._build_plan_episode("plan_fix_test")

    reserve_episodes(
        verified_recovery_episode,
        minimum_conversations["verified_error_recovery"],
        "verified_error_recovery",
    )
    if mode == PAPER_TRAIN_V2_MODE:
        reserve_episodes(
            generator.paper_v2_schema_episode,
            minimum_conversations["paper_v2_schema_trajectories"],
            "paper_v2_schema_trajectories",
        )

    for length, count in sorted(plan_length_minimums.items()):
        names = [
            name
            for name, (_attribute, registered_length) in Generator._PLAN_BUILDERS.items()
            if registered_length == length
        ]
        plan_index = 0

        def plan_episode_for_length(
            plan_names: tuple[str, ...] = tuple(names),
        ) -> Conversation:
            nonlocal plan_index
            which = plan_names[plan_index % len(plan_names)]
            plan_index += 1
            return generator._build_plan_episode(which)

        reserve_episodes(plan_episode_for_length, count, f"plan_length_{length}")

    fill_episodes: list[Conversation] = []
    fill_target = n_multi - len(quota_episodes)
    fill_attempts = 0
    fill_cap = max(10_000, fill_target * 500)
    while len(fill_episodes) < fill_target and fill_attempts < fill_cap:
        fill_attempts += 1
        episode = generator.episodes(1)[0]
        key = episode.to_json()
        if key in quota_episode_keys or contains_holdout_prompt(episode):
            continue
        quota_episode_keys.add(key)
        fill_episodes.append(episode)
    if len(fill_episodes) != fill_target:
        raise RuntimeError(
            "generator diversity exhausted for globally unique multi-turn fill: "
            f"{len(fill_episodes)}/{fill_target} after {fill_attempts} attempts"
        )

    for episode in [*quota_episodes, *fill_episodes]:
        episode.tools = tools
        episode.meta.update({"split": split, "generator": "deterministic_templates"})
        conversations.append(episode)
    if verification.get("rule_based", True):
        conversations = [conv for conv in conversations if valid(conv)]
    if len(conversations) != n_samples:
        raise RuntimeError(f"verification dropped {n_samples - len(conversations)} conversations")
    leaked_holdouts = sorted(
        {
            message.content
            for conversation in conversations
            for message in conversation.messages
            if message.role == Role.user
            and message.content is not None
            and prompt_is_held_out(message.content)
        }
    )
    if leaked_holdouts:  # pragma: no cover - every assembly path filters before this assertion
        raise RuntimeError(
            f"configured exact prompt holdout leaked into synthetic data: {leaked_holdouts[0]!r}"
        )
    rule_verified = bool(verification.get("rule_based", True))
    verification_scope = (
        "canonical_schema_tool_arguments_and_planner_metadata"
        if rule_verified
        else "none"
    )
    if rule_verified and mode == PAPER_TRAIN_V2_MODE:
        verification_scope = (
            "canonical_schema_tool_arguments_planner_metadata_and_paper_v2_slot_audit"
        )
    for conversation in conversations:
        conversation.meta.update(
            {
                # ``verified`` historically appeared on multi-turn template rows. Keep the
                # compatibility field, but make its meaning honest: no tool/environment result is
                # executed by this deterministic generator.
                "verified": False,
                "rule_verified": rule_verified,
                "model_verified": False,
                "environment_executed": False,
                "verification_scope": verification_scope,
            }
        )
        if mode == PAPER_TRAIN_V2_MODE:
            conversation.meta.update(
                {
                    "template_mode": PAPER_TRAIN_V2_MODE,
                    "template_mode_version": PAPER_TRAIN_V2_MODE_VERSION,
                }
            )

    structure_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    argument_value_counts: Counter[str] = Counter()
    plan_length_counts: Counter[int] = Counter()
    schema_property_type_counts: Counter[str] = Counter()
    for tool in tools:
        for property_schema in tool.parameters.get("properties", {}).values():
            property_type = property_schema.get("type")
            if isinstance(property_type, str):
                schema_property_type_counts[property_type] += 1

    def argument_type(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        return "object"

    def response_grounded_followup(conversation: Conversation) -> bool:
        initial_user = next(
            (
                message.content
                for message in conversation.messages
                if message.role == Role.user and message.content
            ),
            "",
        )
        prior_tool_responses: list[str] = []
        for message in conversation.messages:
            if message.role == Role.tool and message.tool_response:
                prior_tool_responses.append(message.tool_response)
            elif message.role == Role.assistant and prior_tool_responses:
                for call in message.tool_calls:
                    for value in call.arguments.values():
                        if (
                            isinstance(value, str)
                            and value
                            and value not in initial_user
                            and any(value in response for response in prior_tool_responses)
                        ):
                            return True
        return False

    def verified_error_recovery(conversation: Conversation) -> bool:
        failure_seen = False
        remediation_seen = False
        retry_seen = False
        for message in conversation.messages:
            if message.role == Role.tool and message.tool_response:
                response = message.tool_response.casefold()
                if "failed" in response:
                    failure_seen = True
                elif failure_seen and retry_seen and "all tests passed" in response:
                    return True
            elif failure_seen and message.role == Role.assistant and message.tool_calls:
                names = {call.name for call in message.tool_calls}
                if any(name != "run_tests" for name in names):
                    remediation_seen = True
                if remediation_seen and "run_tests" in names:
                    retry_seen = True
        return False

    for conversation in conversations:
        assistant_calls = [
            len(message.tool_calls)
            for message in conversation.messages
            if message.role == Role.assistant
        ]
        total_calls = sum(assistant_calls)
        if len(conversation.messages) > 2:
            structure_counts["multi_turn_conversations"] += 1
        elif conversation.meta.get("category") == "no_tool":
            structure_counts["irrelevance_conversations"] += 1
        elif total_calls == 0:
            structure_counts["text_conversations"] += 1
        elif total_calls == 1:
            structure_counts["single_call_conversations"] += 1
        else:
            structure_counts["parallel_call_conversations"] += 1
        structure_counts["assistant_tool_calls"] += total_calls
        if any(count > 1 for count in assistant_calls):
            behavior_counts["parallel_calls"] += 1
        if conversation.meta.get("category") == "no_tool" or (
            conversation.meta.get("kind") == "planner_episode"
            and conversation.meta.get("plan_len") == 0
        ):
            behavior_counts["explicit_restraint"] += 1
        if response_grounded_followup(conversation):
            behavior_counts["tool_response_grounded_followups"] += 1
        if verified_error_recovery(conversation):
            behavior_counts["verified_error_recovery"] += 1
        if conversation.meta.get("kind") == "paper_v2_schema_episode":
            behavior_counts["paper_v2_schema_trajectories"] += 1
        has_integer = False
        has_boolean = False
        has_number = False
        has_enum = False
        has_multiple_arguments = False
        for message in conversation.messages:
            for call in message.tool_calls:
                has_multiple_arguments = has_multiple_arguments or len(call.arguments) > 1
                property_schemas = tool_map[call.name].parameters.get("properties", {})
                for name, value in call.arguments.items():
                    value_type = argument_type(value)
                    argument_value_counts[value_type] += 1
                    has_integer = has_integer or value_type == "integer"
                    has_boolean = has_boolean or value_type == "boolean"
                    has_number = has_number or value_type == "number"
                    argument_schema = property_schemas.get(name, {})
                    has_enum = has_enum or value in argument_schema.get("enum", ())
        if has_integer:
            behavior_counts["integer_arguments"] += 1
        if has_boolean:
            behavior_counts["boolean_arguments"] += 1
        if has_number:
            behavior_counts["number_arguments"] += 1
        if has_enum:
            behavior_counts["enum_arguments"] += 1
        if has_multiple_arguments:
            behavior_counts["multiple_arguments"] += 1
        if conversation.meta.get("kind") == "planner_episode":
            plan_length_counts[int(conversation.meta["plan_len"])] += 1

    for name, minimum in minimum_conversations.items():
        if behavior_counts[name] < minimum:
            raise RuntimeError(
                f"coverage quota not met for {name}: {behavior_counts[name]}/{minimum}"
            )
    for length, minimum in plan_length_minimums.items():
        if plan_length_counts[length] < minimum:
            raise RuntimeError(
                f"plan length {length} quota not met: {plan_length_counts[length]}/{minimum}"
            )

    out = Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(conv.to_json() + "\n" for conv in conversations)
    tmp.replace(out)
    with out.open("rb") as handle:
        output_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

    rule_verification_scope = [
        "canonical_conversation_schema",
        "registered_tool_name",
        "required_argument_presence",
        "primitive_argument_type",
        "argument_enum",
        "planner_metadata_matches_tool_sequence",
    ]
    split_contract: dict[str, object] = {
        "named_slot_pools": ("declared *_TRAIN and *_EVAL value pools are pairwise disjoint"),
        "primitive_value_disjointness_claimed": False,
        "template_disjointness_claimed": False,
        "note": (
            "schema enum values, no-argument intents, and template vocabulary may be shared; "
            "configured suite prompts are excluded only by canonical user-prompt equality"
        ),
    }
    behavior_definitions = {
        "explicit_restraint": (
            "no_tool category or zero-tool planner episode; calling a tool is incorrect"
        ),
        "enum_arguments": "conversation containing an argument admitted by its tool schema enum",
        "integer_arguments": "conversation containing a non-boolean integer argument",
        "multiple_arguments": "conversation containing a call with at least two arguments",
        "parallel_calls": "conversation containing one assistant turn with multiple tool calls",
        "tool_response_grounded_followups": (
            "string argument absent from the initial user turn and copied from a prior "
            "tool response"
        ),
        "verified_error_recovery": (
            "compatibility quota name: rule-audited scripted trace with a FAILED tool "
            "response, remediation call, and later All tests passed response; tool outcomes "
            "are template literals and are not executed"
        ),
    }
    argument_schema_coverage: dict[str, object] = {
        "property_types": dict(sorted(schema_property_type_counts.items())),
        "absent_primitive_types": ["boolean", "number"],
        "note": (
            "STANDARD_TOOLS defines no boolean or number properties; the synthetic corpus "
            "therefore does not claim training coverage for those argument types"
        ),
    }
    complexity_contract: dict[str, object] = {
        "enforced": {"multi_turn": multi_fraction},
        "regular_sampling": "deterministic_maker_registry",
        "quota_sampling": bool(reserved_regular or reserved_multi),
    }
    coverage_contract: dict[str, object] = {
        "minimum_conversations": minimum_conversations,
        "plan_length_minimums": {
            str(length): count for length, count in sorted(plan_length_minimums.items())
        },
        "sampling": "deterministic_reserved_batches_then_registry_fill",
        "semantics": "minimums; deterministic fill may increase realized behavior counts",
    }
    if mode == PAPER_TRAIN_V2_MODE:
        rule_verification_scope.extend(
            [
                "paper_v2_scroll_number_boolean_schema",
                "paper_v2_train_only_slot_pool_disjointness",
            ]
        )
        split_contract["paper_train_v2"] = {
            "mode_version": PAPER_TRAIN_V2_MODE_VERSION,
            "scope": "train-only schema/template enrichment",
            "frozen_eval_slot_audit": slot_audit,
        }
        behavior_definitions.update(
            {
                "boolean_arguments": (
                    "conversation containing a schema-valid non-numeric JSON boolean argument"
                ),
                "number_arguments": (
                    "conversation containing a schema-valid non-integral JSON number argument"
                ),
                "paper_v2_schema_trajectories": (
                    "multi-turn precise-scroll trajectory tagged paper_v2_schema_episode"
                ),
            }
        )
        argument_schema_coverage.update(
            {
                "absent_primitive_types": [],
                "note": (
                    "The versioned paper_train_v2 scroll overlay adds optional boolean and number "
                    "properties; positive quotas require both types in exported tool arguments"
                ),
                "schema_overlay": {
                    "mode": PAPER_TRAIN_V2_MODE,
                    "mode_version": PAPER_TRAIN_V2_MODE_VERSION,
                    "tool": "scroll",
                    "properties": {"amount": "number", "smooth": "boolean"},
                },
            }
        )
        complexity_contract.update(
            {
                "template_mode": PAPER_TRAIN_V2_MODE,
                "template_mode_version": PAPER_TRAIN_V2_MODE_VERSION,
                "strata": [
                    "schema_number_boolean",
                    "tool-like_irrelevance",
                    "scripted_failure_recovery",
                    "multi_turn_schema_use",
                ],
            }
        )
        coverage_contract["paper_train_v2_preflight"] = paper_v2_preflight

    manifest_without_hash = {
        "kind": MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": "localagent.data.schema.Conversation",
        "conversation_serialization": CONVERSATION_SERIALIZATION,
        "generator_config": config_identity.as_dict(),
        "rows": len(conversations),
        "output_bytes": out.stat().st_size,
        "output_sha256": output_sha256,
        "single_turn": n_regular + n_irrelevant,
        "multi_turn": n_multi,
        "irrelevance": n_irrelevant,
        "seed": seed,
        "level": level,
        "split": split,
        "rule_verified": verification.get("rule_based", True),
        "rule_verification_scope": (
            rule_verification_scope
            if verification.get("rule_based", True)
            else []
        ),
        "model_verified": False,
        "environment_executed": False,
        "verification_claim": "rule_audited_not_environment_executed",
        "split_contract": split_contract,
        "exact_prompt_holdouts": {
            "artifacts": holdout_artifacts,
            "artifact_count": len(holdout_artifacts),
            "normalized_unique_prompts": len(holdout_prompts),
            "normalized_prompts_sha256": holdout_prompts_sha256,
            "match_scope": "all Conversation user-message content fields",
            "match_mode": "canonical normalized equality",
            "normalization": {
                "unicode": "NFKC",
                "whitespace": "Unicode split then single ASCII-space join",
                "case": "Unicode casefold",
                "case_sensitive": False,
            },
            "output_matches": 0,
        },
        "structural_counts": dict(sorted(structure_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "behavior_definitions": behavior_definitions,
        "argument_value_counts": dict(sorted(argument_value_counts.items())),
        "argument_schema_coverage": argument_schema_coverage,
        "plan_length_counts": {
            str(length): count for length, count in sorted(plan_length_counts.items())
        },
        "complexity_contract": complexity_contract,
        "coverage_contract": coverage_contract,
    }
    manifest, manifest_payload = self_hashed_manifest(manifest_without_hash)
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_bytes(manifest_payload)
    manifest_tmp.replace(manifest_path)
    print(json.dumps({"out": str(out), **manifest}, indent=2))
