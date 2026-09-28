"""Hand-authored free-form dispatch eval — the honest out-of-distribution test.

These queries are written in natural, varied phrasing that deliberately does NOT follow the synthetic
templates, so accuracy here measures whether tool *selection* generalizes (not template memorization).
Each entry is (query, gold_tool). Arg values are intentionally embedded as literal substrings so the
pointer-copy path is still scorable, but the primary metric this set drives is SELECTION accuracy.
"""

from __future__ import annotations

# (query, gold tool name). Gold reflects the most appropriate single tool in STANDARD_TOOLS.
FREEFORM_EVAL: list[tuple[str, str]] = [
    # --- web_search / knowledge ---
    ("What is the color of a monkey?", "web_search"),
    ("Look up who invented the telephone.", "web_search"),
    ("Google the tallest mountain in Africa.", "web_search"),
    ("Find out when the Eiffel Tower was built.", "web_search"),
    ("Can you search for the best ramen in Tokyo?", "web_search"),
    # --- define ---
    ("What does the word ephemeral mean?", "define"),
    ("Define photosynthesis for me.", "define"),
    ("Give me the definition of entropy.", "define"),
    # --- open_url / http ---
    ("Open https://github.com/pytorch/pytorch in the browser.", "open_url"),
    ("Go to example.com and load the homepage.", "open_url"),
    ("Fetch the JSON from https://api.github.com/repos/torvalds/linux.", "http_request"),
    ("Make a GET request to https://httpbin.org/get.", "http_request"),
    # --- download_file ---
    ("Download the dataset from https://example.com/data.zip.", "download_file"),
    ("Grab the file at https://example.com/report.pdf and save it.", "download_file"),
    # --- get_news ---
    ("Any recent news about electric vehicles?", "get_news"),
    ("What's the latest headlines on climate policy?", "get_news"),
    # --- list_dir / find_files ---
    ("List the files in the src directory.", "list_dir"),
    ("Show me what's inside the build folder.", "list_dir"),
    ("ls the current directory.", "list_dir"),
    ("Find all python files under tests.", "find_files"),
    # --- grep_search ---
    ("Search the codebase for the function train_step.", "grep_search"),
    ("Grep for TODO comments in the repo.", "grep_search"),
    ("Where is the string 'eos_id' used?", "grep_search"),
    # --- make_dir ---
    ("Make a directory called build.", "make_dir"),
    ("Create a new folder named output.", "make_dir"),
    ("mkdir logs please.", "make_dir"),
    # --- run_command / run_python / run_tests ---
    ("Run the test suite.", "run_tests"),
    ("Execute the unit tests.", "run_tests"),
    ("Run python train.py.", "run_command"),
    ("Run this Python: print(2 ** 16).", "run_python"),
    ("Run the command echo hello in the shell.", "run_command"),
    # --- read/write/edit file ---
    ("Read the file data/loader.py.", "read_file"),
    ("Open and show me the contents of README.md.", "read_file"),
    ("Write a config to settings.yaml.", "write_file"),
    # --- git ---
    ("What's the git status of the repo?", "git_status"),
    ("Show me the git diff.", "git_diff"),
    ("Commit the staged changes with message fix bug.", "git_commit"),
    # --- app_action ---
    ("Email Dana the quarterly report.", "send_email"),
    ("Send a slack message to the team channel saying deploy done.", "slack_send"),
    ("Add a calendar event for the standup at 9am.", "calendar_event"),
    ("Remind me to water the plants.", "set_reminder"),
    # --- compute ---
    ("What is 18 * 24?", "calculator"),
    ("Compute 144 divided by 12.", "calculator"),
    # --- weather / music ---
    ("What's the weather like in Oslo?", "get_weather"),
    ("Play Bohemian Rhapsody.", "play_music"),
]


# Hand-authored free-form TRAIN set: natural, NON-templated queries for training the selector /
# route head on the free-form distribution. Same varied register as FREEFORM_EVAL, but DISJOINT
# from it in BOTH phrasing AND slot values (different cities, songs, files, URLs, terms, ...).
# Covers all 50 tools in STANDARD_TOOLS. Each entry is (query, gold_tool).
FREEFORM_TRAIN: list[tuple[str, str]] = [
    # --- web_search / knowledge ---
    ("How many moons does Jupiter have?", "web_search"),
    ("Look up who painted the Mona Lisa.", "web_search"),
    ("I keep forgetting — what year did the Berlin Wall fall?", "web_search"),
    ("Quick, find me the capital of Mongolia.", "web_search"),
    ("Can you check how tall Mount Kilimanjaro is?", "web_search"),
    # --- define ---
    ("What does the term 'serendipitous' actually mean?", "define"),
    ("Define mitochondria.", "define"),
    ("Give me a quick definition of 'recursion'.", "define"),
    # --- open_url / http_request ---
    ("Pull up news.ycombinator.com in my browser.", "open_url"),
    ("Take me to the homepage at wikipedia.org.", "open_url"),
    ("Hit the endpoint at https://api.stripe.com/v1/charges and show the response.", "http_request"),
    ("Send a GET to https://jsonplaceholder.typicode.com/todos/1.", "http_request"),
    # --- download_file ---
    ("Pull down the archive at https://releases.ubuntu.com/jammy.iso for me.", "download_file"),
    ("Save the spreadsheet from https://files.example.org/q3-budget.xlsx to disk.", "download_file"),
    ("Grab https://cdn.fonts.net/inter.woff2 and store it locally.", "download_file"),
    # --- get_news ---
    ("Anything happening with the housing market lately?", "get_news"),
    ("Catch me up on the latest in AI regulation.", "get_news"),
    ("What's the buzz around the upcoming Olympics?", "get_news"),
    # --- list_dir / find_files ---
    ("Show me everything inside the vendor folder.", "list_dir"),
    ("What's sitting in the migrations directory?", "list_dir"),
    ("Track down every .yaml file under config.", "find_files"),
    ("Locate all the test_*.go files in the repo.", "find_files"),
    # --- grep_search ---
    ("Hunt through the source for where compute_loss is called.", "grep_search"),
    ("Find every spot we reference the API_TOKEN constant.", "grep_search"),
    ("Where in the code do we handle the SIGTERM signal?", "grep_search"),
    # --- make_dir ---
    ("Spin up a folder called artifacts for me.", "make_dir"),
    ("I need a new directory named checkpoints.", "make_dir"),
    # --- run_command / run_python / run_tests ---
    ("Kick off the whole test suite, please.", "run_tests"),
    ("Can we run all the unit tests now?", "run_tests"),
    ("Run python manage.py migrate.", "run_command"),
    ("Go ahead and execute python scripts/seed_db.py.", "run_command"),
    ("Fire off du -sh in the terminal.", "run_command"),
    ("Run this Python: print([n*n for n in range(5)]).", "run_python"),
    ("Evaluate sorted({'b':2,'a':1}.items()) in Python for me.", "run_python"),
    # --- read/write/edit file ---
    ("Let me see what's in src/config/loader.ts.", "read_file"),
    ("Cat the file pyproject.toml.", "read_file"),
    ("Drop a new dotenv file at deploy/.env.production.", "write_file"),
    ("Make a fresh file called notes/standup.md.", "write_file"),
    ("Go tweak the function signatures in lib/handlers.go.", "edit_file"),
    ("Patch up the typo in docs/getting-started.md.", "edit_file"),
    # --- git ---
    ("How's the working tree looking right now?", "git_status"),
    ("Show me what I've changed but haven't staged yet.", "git_diff"),
    ("Wrap up these changes with a commit message of tidy logging output.", "git_commit"),
    ("Commit everything as bump dependencies to latest.", "git_commit"),
    # --- apply_patch ---
    ("Apply this diff to the file core/scheduler.py.", "apply_patch"),
    ("Roll the patch into web/components/Header.tsx.", "apply_patch"),
    # --- productivity ---
    ("Shoot Priyanka an email about the budget review.", "send_email"),
    ("Drop a message in the #releases channel saying we shipped v2.", "slack_send"),
    ("Block off my calendar tomorrow for the dentist.", "calendar_event"),
    ("Nudge me later to refill my prescription.", "set_reminder"),
    ("Jot down 'brainstorm Q4 OKRs' in my Notion.", "notion_write"),
    ("File a Jira ticket for the broken password reset flow.", "jira_issue"),
    # --- timers / compute ---
    ("Give me a 25 minute timer for a focus block.", "set_timer"),
    ("How much is 4096 divided by 16?", "calculator"),
    ("What's 27 squared?", "calculator"),
    # --- weather / music ---
    ("Is it going to rain in Reykjavik today?", "get_weather"),
    ("How chilly is it over in Helsinki right now?", "get_weather"),
    ("Throw on some Fleetwood Mac — Dreams.", "play_music"),
    ("Put on Take Five by Dave Brubeck.", "play_music"),
    # --- planner ---
    ("Help me put together a plan to learn the cello.", "planner"),
    ("I want to map out how to migrate us off the monolith.", "planner"),
    # --- sql / env / docker / packages ---
    ("Run SELECT * FROM subscriptions WHERE active = 0 against the db.", "sql_query"),
    ("Query the database with SELECT count(*) FROM page_views.", "sql_query"),
    ("What's the DATABASE_URL set to in the environment?", "env_get"),
    ("Read the env var STRIPE_SECRET_KEY for me.", "env_get"),
    ("Spin up a container from postgres:16.", "docker_run"),
    ("Boot the redis:7-alpine image in Docker.", "docker_run"),
    ("Add the package httpx to the project.", "install_package"),
    ("Pull in pytest-asyncio as a dependency.", "install_package"),
    # --- process / archive ---
    ("Kill the stray gunicorn process.", "kill_process"),
    ("Terminate whatever's running as celery-worker.", "kill_process"),
    ("What processes are eating my CPU right now?", "list_processes"),
    ("Unzip the bundle frontend-dist.tar.gz.", "unzip"),
    ("Extract everything out of logs-2026.zip.", "unzip"),
    # --- computer use ---
    ("Snap a screenshot of whatever's on my screen.", "screenshot"),
    ("Click the 'Submit order' button.", "click"),
    ("Double-click the 'README' file to open it.", "double_click"),
    ("Type my email address jordan@studio.dev into the form.", "type_text"),
    ("Just hit the Escape key.", "key_press"),
    ("Scroll down a bit so I can see the footer.", "scroll"),
    ("Drag the 'Inbox' card over onto the 'Done' column.", "drag"),
    ("Hold on for about 5 seconds before continuing.", "wait"),
    ("Move the cursor over to the 'Settings' gear.", "move_cursor"),
    ("Fire up the Calculator app.", "open_app"),
    ("What's sitting in my clipboard at the moment?", "read_clipboard"),
    ("Copy the order number 99213 to my clipboard.", "write_clipboard"),
]
