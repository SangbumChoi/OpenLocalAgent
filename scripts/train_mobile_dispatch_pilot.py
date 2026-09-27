#!/usr/bin/env python
"""Train an additive standard+mobile dispatch selector on a verified mobile Conversation mix.

The LM backbone and legacy 50-tool head remain shape-compatible.  Route and dense-selector probes
are continued, with an optional train-only pointer/copy-head continuation for literal arguments.
The exported tool pool is explicitly recorded.  A deterministic episode holdout reports selection
and grounded-call metrics; it is not an Android emulator score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
from localagent.agent.pointer_head import PTR_ARGS, PointerHead, gold_span
from localagent.agent.routes import ROUTE_INDEX, ROUTES, RouteHead, route_of
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.agent_synth import Generator, Sample
from localagent.data.paraphrase import paraphrase_samples
from localagent.data.schema import Conversation, Role
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ASSISTANT, USER, load_tokenizer


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _load_rows(paths: list[Path]) -> list[Conversation]:
    rows: list[Conversation] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        raw = json.loads(line)
                        conversation = Conversation.from_json(line)
                        metadata = raw.get("meta", {})
                        if isinstance(metadata, dict):
                            conversation.meta = dict(metadata)
                        conversation.meta.setdefault("record_id", raw.get("record_id", "unknown"))
                        rows.append(conversation)
                    except Exception as error:  # pragma: no cover - diagnostic context
                        raise ValueError(f"invalid Conversation at {path}:{line_number}") from error
    if not rows:
        raise ValueError("no conversations found")
    return rows


def _mobile_samples(rows: list[Conversation]) -> list[tuple[Sample, str]]:
    samples: list[tuple[Sample, str]] = []
    for row in rows:
        quality = row.meta.get("quality", {})
        episode_value = quality.get("source_episode_id") if isinstance(quality, dict) else None
        if episode_value is None:
            episode_value = row.meta.get("parent_record_id")
        if episode_value is None:
            # The JSON mirror is one screenshot/action per row rather than an episode.  Keep each
            # source row independent so a train/eval file boundary cannot collapse to "unknown".
            episode_value = row.meta.get("source_row_index", "unknown")
        episode = str(episode_value)
        messages = row.messages
        for index, message in enumerate(messages):
            if message.role != Role.assistant or not message.tool_calls:
                continue
            call = message.tool_calls[0]
            if not call.name.startswith("mobile_"):
                continue
            prompt = ""
            for previous in reversed(messages[:index]):
                if previous.role == Role.user and previous.content:
                    prompt = previous.content
                    break
            if not prompt:
                continue
            sample = Sample(
                "mobile",
                "mobile",
                prompt,
                "tool",
                json.dumps(
                    {"arguments": call.arguments, "name": call.name},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                call.name,
                json.dumps(call.arguments, sort_keys=True, separators=(",", ":")),
            )
            samples.append((sample, episode))
    if not samples:
        raise ValueError("no mobile tool turns found in the supplied Conversations")
    return samples


def _productivity_samples() -> list[Sample]:
    """Return deterministic full-field email/Notion prompts with a disjoint held-out tail."""

    rows = [
        ("Create a Notion page titled 'Sprint plan' with content 'Ship the browser pilot'.",
         "notion_create_page", {"title": "Sprint plan", "content": "Ship the browser pilot"}),
        ("Add a Notion page called 'Reading list' containing 'Review the agent papers'.",
         "notion_create_page", {"title": "Reading list", "content": "Review the agent papers"}),
        ("Save a Notion page titled 'Release notes' with content 'WebGPU fallback is documented'.",
         "notion_create_page", {"title": "Release notes", "content": "WebGPU fallback is documented"}),
        ("Create a Notion note named 'Meeting prep' containing 'Ask about emulator access'.",
         "notion_create_page", {"title": "Meeting prep", "content": "Ask about emulator access"}),
        ("Send an email to 'bob@example.com' with subject 'Build status' and body 'The tests pass'.",
         "email_send", {"to": "bob@example.com", "subject": "Build status", "body": "The tests pass"}),
        ("Email 'carol@example.com' about 'Dataset review' saying 'Please check the split'.",
         "email_send", {"to": "carol@example.com", "subject": "Dataset review", "body": "Please check the split"}),
        ("Compose a message to 'david@example.com' titled 'Receipt' with text 'The export is ready'.",
         "email_send", {"to": "david@example.com", "subject": "Receipt", "body": "The export is ready"}),
        ("Send 'eve@example.com' an email with subject 'WebGPU' and body 'The bundle is local'.",
         "email_send", {"to": "eve@example.com", "subject": "WebGPU", "body": "The bundle is local"}),
        ("Write a Notion page titled 'Training plan' with content 'Run the held-out evaluation'.",
         "notion_create_page", {"title": "Training plan", "content": "Run the held-out evaluation"}),
        ("Create a Notion page called 'Mobile notes' containing 'Keep the screen text compact'.",
         "notion_create_page", {"title": "Mobile notes", "content": "Keep the screen text compact"}),
        ("Email 'frank@example.com' with subject 'Pilot' and body 'The mobile loop passed'.",
         "email_send", {"to": "frank@example.com", "subject": "Pilot", "body": "The mobile loop passed"}),
        ("Send a message to 'grace@example.com' titled 'Next step' saying 'Run the real emulator'.",
         "email_send", {"to": "grace@example.com", "subject": "Next step", "body": "Run the real emulator"}),
        # Held-out rows: values and phrasings do not occur in the training prefix.
        ("Make a Notion page titled 'Workshop checklist' whose content is 'Verify every receipt'.",
         "notion_create_page", {"title": "Workshop checklist", "content": "Verify every receipt"}),
        ("Add a Notion page named 'Ablations' with the text 'Compare no-guard dispatch'.",
         "notion_create_page", {"title": "Ablations", "content": "Compare no-guard dispatch"}),
        ("Send an email to 'heidi@example.com' with subject 'Acceptance' and body 'Please review the gate'.",
         "email_send", {"to": "heidi@example.com", "subject": "Acceptance", "body": "Please review the gate"}),
        ("Write to 'ivan@example.com' with the subject 'Mobile result' and message 'Selector accuracy is pending'.",
         "email_send", {"to": "ivan@example.com", "subject": "Mobile result", "body": "Selector accuracy is pending"}),
    ]
    samples = []
    for prompt, name, arguments in rows:
        encoded_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        samples.append(
            Sample(
                "realistic_productivity",
                "app_action",
                prompt,
                "tool",
                json.dumps({"arguments": arguments, "name": name}, sort_keys=True, separators=(",", ":")),
                name,
                encoded_args,
            )
        )
    return samples


def _compact_mobile_samples(mobile: list[tuple[Sample, str]]) -> list[Sample]:
    """Add instruction-only views without changing the held-out raw screen observations."""

    compact: list[Sample] = []
    for sample, _episode in mobile:
        marker = " instruction:"
        prompt = sample.prompt.split(marker, 1)[-1].strip() if marker in sample.prompt else sample.prompt
        if prompt == sample.prompt:
            continue
        compact.append(
            Sample(
                sample.category,
                sample.group,
                prompt,
                sample.kind,
                sample.target,
                sample.ref_name,
                sample.ref_args,
                sample.calls,
            )
        )
    return compact


def _synthetic_mobile_samples() -> list[Sample]:
    """Return balanced, instruction-only mobile dispatch paraphrases.

    Real AndroidControl/AITW rows are valuable for screen grounding but are strongly imbalanced
    toward taps.  These rows deliberately contain no screenshots, task IDs, or held-out app names;
    they teach the selector the action vocabulary while keeping the real environment rows as the
    evidence source for evaluation.
    """

    rows = {
        "mobile_open_app": [
            "On the Android phone, launch the Mail app.",
            "Open the Gmail app on the Android phone.",
            "Launch Calendar on the mobile device.",
            "Start the Maps application on the phone.",
            "Bring up the Camera app on Android.",
            "Open the music app on the handset.",
            "Launch the shopping application on the phone.",
            "Start the browser app on the mobile screen.",
            "Open Settings on the Android device.",
            "Launch Notes on the Android phone.",
            "Open the notes app on the mobile device.",
        ],
        "mobile_click": [
            "Tap Compose at x=80 y=180 on the Android screen.",
            "Tap the Compose button at x=120 y=220 on the phone.",
            "Click the Search icon at x=42 y=90 on the mobile screen.",
            "Touch the Settings tile on the Android device.",
            "Select the first result on the phone screen.",
            "Tap the blue action button on the handset.",
            "Click the menu item shown on the mobile display.",
            "Touch the checkbox at x=300 y=500.",
            "Tap the notification card on Android.",
            "Tap New folder in the Notes app at x=120 y=180.",
            "Tap the first todo row in Notes at x=160 y=260.",
            "Select the New note control in the mobile notes app.",
        ],
        "mobile_long_press": [
            "Long-press the photo at x=160 y=340 on the phone.",
            "Hold the conversation row on the mobile screen.",
            "Press and hold the app icon on Android.",
            "Long press the selected message on the handset.",
            "Hold the list item at x=240 y=600.",
            "Keep your finger down on the phone's document row.",
            "Long-press the calendar event on the mobile device.",
            "Hold the video thumbnail on the Android screen.",
        ],
        "mobile_scroll": [
            "On Android, scroll down to older notifications.",
            "Scroll down on the phone to see older items.",
            "Swipe the mobile list upward to reveal more content.",
            "Scroll up on the Android screen.",
            "Move down through the handset's page.",
            "Scroll the phone view toward the bottom.",
            "Scroll upward through the mobile results.",
            "Move the Android page down to the next section.",
            "Scroll back toward the top of the phone screen.",
        ],
        "mobile_swipe": [
            "On mobile, swipe from x=90 y=640 to x=90 y=280.",
            "Swipe from x=100 y=700 to x=100 y=300 on the phone.",
            "Drag a finger from the bottom toward the top of the mobile display.",
            "Swipe left across the Android screen.",
            "Swipe right from x=80 y=400 to x=500 y=400.",
            "Make an upward swipe on the handset.",
            "Move the phone screen with a downward swipe.",
            "Swipe across the mobile carousel from left to right.",
            "Perform a diagonal swipe on Android.",
        ],
        "mobile_input_text": [
            "On the phone, type 'contact@example.org' into the focused address field.",
            "Type 'hello@example.com' into the focused phone field.",
            "Enter 'conference notes' in the mobile text box.",
            "Input 'WebGPU pilot' on the Android keyboard.",
            "Fill the focused handset field with 'search term'.",
            "Type the code 'A7B9' into the phone input.",
            "Enter a message reading 'see you soon' on mobile.",
            "Write '42' in the active Android field.",
            "Input the requested text into the focused phone control.",
            "Type 'weekly plan' into the focused Notes title field.",
            "Enter 'buy groceries' in the active mobile note field.",
            "Write 'remember the meeting' into the Notes editor on the phone.",
        ],
        "mobile_navigate_home": [
            "Go to the home screen on the Android phone.",
            "Return to the mobile device home screen.",
            "Press Home on the handset.",
            "Navigate to the phone's home screen.",
            "Leave the current app and return home on Android.",
            "Use the Home action on the mobile device.",
            "Back out to the Android launcher home.",
            "Show the phone home screen.",
        ],
        "mobile_navigate_back": [
            "On the Android phone, go back from the current window.",
            "Go back to the previous page on the phone.",
            "Navigate back on the Android device.",
            "Return to the prior mobile screen.",
            "Press Back on the handset.",
            "Back out of the current phone window.",
            "Use the Android back action.",
            "Return to the previous view on mobile.",
            "Leave this screen using Back.",
            "Go back from the mobile note editor.",
        ],
        "mobile_press_enter": [
            "Press Enter on the Android keyboard.",
            "Submit the focused mobile field with the Enter key.",
            "Hit Return on the phone.",
            "Use the handset's ENTER action.",
            "Confirm the text input with Enter on mobile.",
            "Press the enter key on Android.",
            "Send the focused form with Return on the phone.",
            "Activate the mobile keyboard's Enter button.",
            "Press Enter to save the focused note field.",
            "Confirm the Notes dialog with the phone Enter key.",
        ],
        "mobile_wait": [
            "Wait for the phone screen to finish loading.",
            "Pause until the Android app responds.",
            "Wait on the mobile device.",
            "Sleep briefly for the handset UI to settle.",
            "Wait for the next phone state.",
            "Pause while the mobile page loads.",
            "Allow the Android screen to update before continuing.",
            "Wait a moment on the phone.",
        ],
        "mobile_submit_answer": [
            "Submit the mobile task answer 'alpha, beta'.",
            "On the phone, send the answer 'one; two' for the current task.",
            "提交手机任务答案「会议、购物清单」。",
            "把当前手机任务的答案提交为「第一项，第二项」。",
            "Answer the handset question with 'red, blue'.",
            "Submit the requested phone answer '完成'.",
            "在移动设备上提交文本答案「已确认」。",
            "Send the current mobile answer '7'.",
        ],
    }
    answer_values = {
        "Submit the mobile task answer 'alpha, beta'.": "alpha, beta",
        "On the phone, send the answer 'one; two' for the current task.": "one; two",
        "提交手机任务答案「会议、购物清单」。": "会议、购物清单",
        "把当前手机任务的答案提交为「第一项，第二项」。": "第一项，第二项",
        "Answer the handset question with 'red, blue'.": "red, blue",
        "Submit the requested phone answer '完成'.": "完成",
        "在移动设备上提交文本答案「已确认」。": "已确认",
        "Send the current mobile answer '7'.": "7",
    }
    return [
        Sample(
            "realistic_mobile_synthetic",
            "mobile",
            prompt,
            "tool",
            json.dumps(
                {"arguments": ({"message": answer_values[prompt]} if name == "mobile_submit_answer" else {}),
                 "name": name},
                sort_keys=True,
                separators=(",", ":"),
            ),
            name,
            json.dumps(
                {"message": answer_values[prompt]} if name == "mobile_submit_answer" else {},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for name, prompts in rows.items()
        for prompt in prompts
    ]


def _state_conditioned_trajectory_samples() -> list[Sample]:
    """Return disjoint state-conditioned trajectory prompts for dispatch adaptation.

    These examples teach the selector to ignore serialized observation fields and attend to the
    requested next action.  Values, app names, and phrasing are intentionally different from the
    browser trajectory holdout in ``spaces/localagent-webgpu/mobile-trajectories.js``.
    """

    rows = [
        (
            'Goal: Open Calendar. Current state JSON: {"app":"home","screen":"home"}. '
            "Next required action: Launch Calendar on the Android phone.",
            "mobile_open_app",
            {"app_name": "Calendar"},
        ),
        (
            'Goal: open a message. Current state JSON: {"app":"Mail","screen":"inbox"}. '
            "Next required action: Tap the first message at x=64 y=148.",
            "mobile_click",
            {"x": 64, "y": 148},
        ),
        (
            'Goal: fill a form. Current state JSON: {"screen":"compose","focus":"to"}. '
            "Next required action: Type 'team@example.org' into the recipient field.",
            "mobile_input_text",
            {"text": "team@example.org"},
        ),
        (
            'Goal: browse older items. Current state JSON: {"screen":"inbox","scroll":0}. '
            "Next required action: Scroll down on the Android screen.",
            "mobile_scroll",
            {"direction": "down"},
        ),
        (
            'Goal: return home. Current state JSON: {"app":"Maps","screen":"details"}. '
            "Next required action: Go to the Android home screen.",
            "mobile_navigate_home",
            {},
        ),
        (
            'Goal: send a status update. Current state JSON: {"app":"Mail","screen":"compose",'
            '"email":{"to":"team@example.org","subject":"Status","body":"Ready"}}. '
            "Next required action: Send the message.",
            "email_send",
            {"to": "team@example.org", "subject": "Status", "body": "Ready"},
        ),
        (
            'Goal: capture a note. Current state JSON: {"app":"Notes","screen":"home"}. '
            "Next required action: Create a Notion page titled 'Launch log' with content 'QA passed'.",
            "notion_create_page",
            {"title": "Launch log", "content": "QA passed"},
        ),
        (
            'Goal: inspect the dashboard. Current state JSON: {"page":null}. '
            "Next required action: Open https://example.local/dashboard in the browser.",
            "open_url",
            {"url": "https://example.local/dashboard"},
        ),
        (
            'Goal: inspect the dashboard. Current state JSON: {"page":"dashboard","focused":null}. '
            "Next required action: Click the Refresh button.",
            "click",
            {"target": "the Refresh button"},
        ),
        (
            'Goal: search the dashboard. Current state JSON: {"focused":"search"}. '
            "Next required action: Type 'build 42' into the focused search field.",
            "type_text",
            {"text": "build 42"},
        ),
        (
            'Goal: submit the dashboard search. Current state JSON: {"query":"build 42"}. '
            "Next required action: Press Enter.",
            "key_press",
            {"key": "Enter"},
        ),
    ]
    # Balance the small browser slice so the 62-tool candidate pool does not turn every
    # state-conditioned web action into a frequent mobile distractor.
    rows.extend([
        (
            'Goal: inspect the inbox. Current state JSON: {"page":null}. '
            "Next required action: Navigate to https://example.local/inbox.",
            "open_url", {"url": "https://example.local/inbox"},
        ),
        (
            'Goal: review the calendar. Current state JSON: {"page":null}. '
            "Next required action: Open https://example.local/calendar in the browser.",
            "open_url", {"url": "https://example.local/calendar"},
        ),
        (
            'Goal: inspect a report. Current state JSON: {"page":null}. '
            "Next required action: Visit https://example.local/reports/weekly.",
            "open_url", {"url": "https://example.local/reports/weekly"},
        ),
        (
            'Goal: find an invoice. Current state JSON: {"page":"inbox","focused":null}. '
            "Next required action: Click the 'Search mail' field.",
            "click", {"target": "the Search mail field"},
        ),
        (
            'Goal: open a report. Current state JSON: {"page":"reports","focused":null}. '
            "Next required action: Click the first report result.",
            "click", {"target": "the first report result"},
        ),
        (
            'Goal: refresh a page. Current state JSON: {"page":"dashboard","focused":null}. '
            "Next required action: Click the Refresh control.",
            "click", {"target": "the Refresh control"},
        ),
        (
            'Goal: search invoices. Current state JSON: {"focused":"search"}. '
            "Next required action: Type 'invoice 2026' into the search field.",
            "type_text", {"text": "invoice 2026"},
        ),
        (
            'Goal: filter the report. Current state JSON: {"focused":"filter"}. '
            "Next required action: Type 'overdue' into the focused filter.",
            "type_text", {"text": "overdue"},
        ),
        (
            'Goal: search contacts. Current state JSON: {"focused":"search"}. '
            "Next required action: Enter 'alice' in the focused search box.",
            "type_text", {"text": "alice"},
        ),
        (
            'Goal: submit the invoice search. Current state JSON: {"query":"invoice 2026"}. '
            "Next required action: Press Enter to submit the search.",
            "key_press", {"key": "Enter"},
        ),
        (
            'Goal: confirm a dialog. Current state JSON: {"dialog":"confirm"}. '
            "Next required action: Press Enter to confirm.",
            "key_press", {"key": "Enter"},
        ),
        (
            'Goal: search contacts. Current state JSON: {"query":"alice"}. '
            "Next required action: Press Enter and load the results.",
            "key_press", {"key": "Enter"},
        ),
        (
            'Goal: create a note folder. Current state JSON: {"app":"notes","screen":"home"}. '
            'Visible text: "New folder" "New note" "Todos". '
            "Next required action: Tap New folder at x=120 y=180.",
            "mobile_click", {"x": 120, "y": 180},
        ),
        (
            'Goal: write a note. Current state JSON: {"app":"notes","screen":"home"}. '
            'Visible text: "New note" "Folders". '
            "Next required action: Tap New note at x=210 y=180.",
            "mobile_click", {"x": 210, "y": 180},
        ),
        (
            'Goal: remove a todo. Current state JSON: {"app":"notes","screen":"todos"}. '
            'Visible text: "Todo A" "Todo B" "Completed". '
            "Next required action: Tap the first todo row at x=150 y=260.",
            "mobile_click", {"x": 150, "y": 260},
        ),
        (
            'Goal: edit a note. Current state JSON: {"app":"notes","screen":"editor","focus":"title"}. '
            'Visible text: "Title" "Content" "Save". '
            "Next required action: Type 'weekly plan' into the focused title field.",
            "mobile_input_text", {"text": "weekly plan"},
        ),
        (
            'Goal: edit a note. Current state JSON: {"app":"notes","screen":"editor","focus":"content"}. '
            'Visible text: "Title" "Content" "Save". '
            "Next required action: Type 'buy groceries' into the focused content field.",
            "mobile_input_text", {"text": "buy groceries"},
        ),
        (
            'Goal: save a note. Current state JSON: {"app":"notes","screen":"editor","focus":"content"}. '
            'Visible text: "Save" "Cancel". '
            "Next required action: Press Enter to save the note.",
            "mobile_press_enter", {"key": "Enter"},
        ),
        (
            'Goal: leave the note editor. Current state JSON: {"app":"notes","screen":"editor"}. '
            'Visible text: "Back" "Save". '
            "Next required action: Navigate back to the Notes list.",
            "mobile_navigate_back", {},
        ),
        (
            'Goal: open notes. Current state JSON: {"app":"home","screen":"launcher"}. '
            'Visible text: "Notes" "Calendar" "Settings". '
            "Next required action: Launch Notes on the phone.",
            "mobile_open_app", {"app_name": "Notes"},
        ),
    ])
    return [
        Sample(
            "realistic_trajectory_synthetic",
            "trajectory",
            prompt,
            "tool",
            json.dumps({"arguments": arguments, "name": name}, sort_keys=True, separators=(",", ":")),
            name,
            json.dumps(arguments, sort_keys=True, separators=(",", ":")),
        )
        for prompt, name, arguments in rows
    ]


def _feature(model: LocalAgentLM, tok, prompt: str, device: str) -> torch.Tensor:
    from localagent.agent.tool_head import _feat

    return _feat(model, tok, prompt, device)


@torch.no_grad()
def _features(
    model: LocalAgentLM,
    tok,
    prompts: list[str],
    device: str,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract final prompt features in batches; the old scalar loop made large public slices unusable."""

    if not prompts:
        raise ValueError("cannot extract features for an empty prompt list")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = [tok.encode(f"{USER}{prompt}{ASSISTANT}")[-model.cfg.max_seq_len :] for prompt in batch]
        width = max(len(row) for row in encoded)
        inputs = torch.full((len(encoded), width), tok.pad_id, dtype=torch.long, device=device)
        lengths = []
        for index, row in enumerate(encoded):
            inputs[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            lengths.append(len(row) - 1)
        _, hidden = model(inputs, return_hidden=True)
        output.append(torch.stack([hidden[index, position] for index, position in enumerate(lengths)]))
    return torch.cat(output, dim=0)


def _checkpoint_tokenizer(parent: dict):
    """Load the tokenizer recorded by the checkpoint, never silently falling back to bytes."""

    metadata = parent.get("tokenizer") or {"kind": "byte"}
    kind = str(metadata.get("kind", "byte")) if isinstance(metadata, dict) else "byte"
    path = metadata.get("path") if isinstance(metadata, dict) else None
    if kind == "bpe" and path is None:
        raise ValueError("BPE checkpoint is missing tokenizer.path")
    return load_tokenizer(kind, path)


def _warm_pointer(parent: dict, d_model: int) -> PointerHead:
    """Load the parent's copy head without changing its argument vocabulary."""

    args = tuple(parent.get("ptr_args", PTR_ARGS))
    pointer = PointerHead(d_model, args=args)
    if parent.get("ptr_head"):
        pointer.load_state_dict(parent["ptr_head"])
    return pointer


def _pointer_examples(
    model: LocalAgentLM,
    tok,
    rows: list[Sample],
    pointer: PointerHead,
    device: str,
) -> list[tuple[torch.Tensor, int, int, int]]:
    examples: list[tuple[torch.Tensor, int, int, int]] = []
    model.eval()
    with torch.no_grad():
        for sample in rows:
            if sample.kind != "tool":
                continue
            try:
                arguments = json.loads(sample.ref_args)
            except json.JSONDecodeError:
                continue
            ids = tok.encode(f"{USER}{sample.prompt}{ASSISTANT}")
            if len(ids) > model.cfg.max_seq_len:
                ids = ids[-model.cfg.max_seq_len :]
            _, hidden = model(
                torch.tensor([ids], dtype=torch.long, device=device),
                return_hidden=True,
            )
            for name, value in arguments.items():
                if name not in pointer.arg_idx or not isinstance(value, str):
                    continue
                span = gold_span(ids, tok.encode(value))
                if span is not None:
                    examples.append((hidden[0, : len(ids)].detach(), pointer.arg_idx[name], span[0], span[1]))
    return examples


def _train_pointer(
    model: LocalAgentLM,
    tok,
    rows: list[Sample],
    pointer: PointerHead,
    *,
    steps: int,
    seed: int,
    device: str,
) -> dict[str, int]:
    """Continue only the copy head on literal train-side argument spans."""

    examples = _pointer_examples(model, tok, rows, pointer, device)
    if not examples:
        raise ValueError("mobile pointer training found no literal string spans")
    optimizer = torch.optim.AdamW(pointer.parameters(), lr=2e-2)
    rng = random.Random(seed)
    pointer.train()
    for _ in range(max(1, steps)):
        batch = [examples[rng.randrange(len(examples))] for _ in range(min(16, len(examples)))]
        loss = torch.zeros((), device=device)
        for hidden, arg_index, start, end in batch:
            arg = torch.tensor([arg_index], dtype=torch.long, device=device)
            start_logits, end_logits = pointer.logits(hidden.unsqueeze(0), arg)
            loss = loss + F.cross_entropy(start_logits, torch.tensor([start], device=device))
            loss = loss + F.cross_entropy(end_logits, torch.tensor([end], device=device))
        optimizer.zero_grad(set_to_none=True)
        (loss / len(batch)).backward()
        optimizer.step()
    pointer.eval()
    return {"examples": len(examples), "steps": max(1, steps)}


@torch.no_grad()
def _pointer_score(
    model: LocalAgentLM,
    tok,
    pointer: PointerHead,
    rows: list[Sample],
    device: str,
) -> dict[str, int | float]:
    total = exact = 0
    for hidden, arg_index, start, end in _pointer_examples(model, tok, rows, pointer, device):
        arg = pointer.args[arg_index]
        predicted = pointer.predict_span(hidden, arg)
        total += 1
        exact += int(predicted == (start, end))
    return {"rows": total, "exact": exact, "exact_rate": exact / max(1, total)}


def _train_probe(
    model: LocalAgentLM,
    tok,
    parent: dict,
    tools,
    mobile: list[tuple[Sample, str]],
    productivity: list[Sample],
    synthetic_mobile: list[Sample],
    trajectory: list[Sample],
    *,
    steps: int,
    device: str,
    seed: int,
    probe_init: str,
    focus_tools: set[str],
    focus_repeat: int,
) -> tuple[RouteHead, DenseToolSelector, dict[str, list[str]]]:
    if probe_init not in {"parent_heads", "random"}:
        raise ValueError("probe_init must be 'parent_heads' or 'random'")
    random.seed(seed)
    torch.manual_seed(seed)
    standard = Generator(level=3, seed=seed, split="train").generate_balanced(3)
    standard += paraphrase_samples(2, seed=seed, split="train")
    mobile_samples = [sample for sample, _ in mobile]
    compact_mobile = _compact_mobile_samples(mobile)
    # Keep the broad standard catalog, but repeat the two deployment-critical productivity tools
    # enough that they are not drowned out by 50 legacy tools and 100+ mobile turns.
    productivity_augmented = productivity * 4
    synthetic_mobile_augmented = synthetic_mobile * 4
    trajectory_augmented = trajectory * 4
    if focus_tools and focus_repeat:
        # The public AndroidControl train split has no mobile_navigate_home rows and only a tiny
        # long-press slice.  Add deterministic synthetic/state-conditioned views for explicitly
        # requested actions, while keeping the public held-out file entirely outside optimization.
        focused_synthetic = [sample for sample in synthetic_mobile if sample.ref_name in focus_tools]
        focused_trajectory = [sample for sample in trajectory if sample.ref_name in focus_tools]
        synthetic_mobile_augmented += focused_synthetic * focus_repeat
        trajectory_augmented += focused_trajectory * focus_repeat
    pool = (
        standard
        + mobile_samples
        + compact_mobile
        + synthetic_mobile_augmented
        + trajectory_augmented
        + productivity_augmented
    )
    examples: dict[str, list[str]] = {}
    for name, values in (parent.get("examples") or {}).items():
        # Older dispatch children stored full accessibility dumps in the example centroid.  Keep
        # standard examples, but compact any inherited mobile rows before they reach either the
        # dense tool tower or the exported retrieval matrix.
        if name.startswith("mobile_"):
            compact_values = [
                value.rsplit(" instruction:", 1)[-1].strip()
                for value in values
                if " instruction:" in value
            ]
            if compact_values:
                examples[name] = compact_values
        else:
            examples[name] = list(values)
    # Keep long screen dumps in the training pool, but use only compact action text for the
    # tool-tower centroid.  Otherwise thousands of accessibility tokens drown out "open", "back",
    # and "wait" in the fixed character n-gram embedding.
    for sample in compact_mobile:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    for sample in synthetic_mobile:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    for sample in trajectory:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    for sample in productivity_augmented:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    route_rows = [(sample, route_of(sample.ref_name)) for sample in pool]
    route_features = _features(model, tok, [sample.prompt for sample, _ in route_rows], device)
    route_labels = torch.tensor([ROUTE_INDEX[label] for _, label in route_rows], device=device)
    route = RouteHead(model.cfg.d_model).to(device)
    if probe_init == "parent_heads" and parent.get("route_head"):
        route.load_state_dict(parent["route_head"])
    route_opt = torch.optim.AdamW(route.parameters(), lr=5e-3)

    name_to_index = {tool.name: index for index, tool in enumerate(tools)}
    selector_rows = [(sample, name_to_index[sample.ref_name]) for sample in pool if sample.ref_name in name_to_index]
    selector_features = _features(
        model, tok, [sample.prompt for sample, _ in selector_rows], device
    )
    selector_labels = torch.tensor([label for _, label in selector_rows], device=device)
    embs = tool_embeddings(tools, device=device, examples=examples)
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=embs.shape[1],
        proj=int(parent.get("selector_proj", 256)),
    ).to(device)
    if probe_init == "parent_heads" and parent.get("dense_selector"):
        selector.load_state_dict(parent["dense_selector"])
    selector_opt = torch.optim.AdamW(selector.parameters(), lr=5e-3)
    rng = random.Random(seed)
    for step in range(max(1, steps)):
        route_idx = torch.tensor([rng.randrange(len(route_rows)) for _ in range(min(64, len(route_rows)))])
        route_loss = F.cross_entropy(route(route_features[route_idx]), route_labels[route_idx])
        route_opt.zero_grad(set_to_none=True)
        route_loss.backward()
        route_opt.step()

        selector_idx = torch.tensor(
            [rng.randrange(len(selector_rows)) for _ in range(min(64, len(selector_rows)))]
        )
        selector_loss = F.cross_entropy(
            selector(selector_features[selector_idx], embs), selector_labels[selector_idx]
        )
        selector_opt.zero_grad(set_to_none=True)
        selector_loss.backward()
        selector_opt.step()
    route.eval()
    selector.eval()
    return route, selector, examples


@torch.no_grad()
def _score(
    model: LocalAgentLM,
    tok,
    route: RouteHead,
    selector: BoundSelector,
    rows: list[tuple[Sample, str]],
    device: str,
) -> dict[str, float | int]:
    route_ok = selector_ok = 0
    breakdown: dict[str, dict[str, int]] = {}
    features = _features(model, tok, [sample.prompt for sample, _ in rows], device)
    for feature, (sample, _) in zip(features, rows):
        route_match = ROUTES[int(route(feature).argmax(-1))] == route_of(sample.ref_name)
        selector_match = selector.rank(feature)[0] == sample.ref_name
        route_ok += route_match
        selector_ok += selector_match
        stats = breakdown.setdefault(sample.ref_name, {"rows": 0, "route_correct": 0, "selector_correct": 0})
        stats["rows"] += 1
        stats["route_correct"] += int(route_match)
        stats["selector_correct"] += int(selector_match)
    total = len(rows)
    return {
        "rows": total,
        "route_accuracy": route_ok / max(1, total),
        "selector_top1": selector_ok / max(1, total),
        "by_tool": {
            name: {
                **stats,
                "route_accuracy": stats["route_correct"] / max(1, stats["rows"]),
                "selector_top1": stats["selector_correct"] / max(1, stats["rows"]),
            }
            for name, stats in sorted(breakdown.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument(
        "--eval-data",
        type=Path,
        action="append",
        default=[],
        help="optional held-out mobile JSONL; never included in probe optimization",
    )
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--pointer-steps",
        type=int,
        default=0,
        help="optional train-only copy-head updates for literal string arguments",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--probe-init",
        choices=("parent_heads", "random"),
        default="parent_heads",
        help="initialize route/selector probes from the parent checkpoint or a seeded reset",
    )
    parser.add_argument(
        "--focus-tool",
        action="append",
        default=[],
        help="synthetic/state rows to oversample (repeat --focus-tool for multiple actions)",
    )
    parser.add_argument(
        "--focus-repeat",
        type=int,
        default=0,
        help="additional copies of focused synthetic/state rows; zero preserves the baseline",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.pointer_steps < 0:
        raise ValueError("pointer-steps must be non-negative")
    if args.focus_repeat < 0:
        raise ValueError("focus-repeat must be non-negative")

    parent = torch.load(args.init, map_location="cpu")
    cfg = ModelConfig(**parent["cfg"])
    model = LocalAgentLM(cfg).to(args.device).eval()
    model.load_state_dict(parent["state_dict"])
    tok = _checkpoint_tokenizer(parent)
    rows = _load_rows(args.data)
    eval_rows = _load_rows(args.eval_data) if args.eval_data else []
    mobile = _mobile_samples(rows)
    if eval_rows:
        train_mobile = mobile
        held_mobile = _mobile_samples(eval_rows)
        episode_ids = sorted({episode for _, episode in mobile})
        holdout_ids = sorted({episode for _, episode in held_mobile})
        split_mode = "external_eval_files"
    else:
        episode_ids = sorted({episode for _, episode in mobile})
        holdout_ids = episode_ids[-max(1, len(episode_ids) // 5) :]
        train_mobile = [item for item in mobile if item[1] not in holdout_ids]
        held_mobile = [item for item in mobile if item[1] in holdout_ids]
        split_mode = "deterministic_episode_tail"
    tools = list(STANDARD_TOOLS) + mobile_tools() + realistic_productivity_tools()
    unknown_focus_tools = sorted(set(args.focus_tool) - {tool.name for tool in tools})
    if unknown_focus_tools:
        raise ValueError(f"unknown focus tool(s): {', '.join(unknown_focus_tools)}")
    productivity = _productivity_samples()
    productivity_train = productivity[:-4]
    productivity_holdout = productivity[-4:]
    synthetic_mobile = _synthetic_mobile_samples()
    trajectory = _state_conditioned_trajectory_samples()
    retrieval_examples: dict[str, list[str]] = {}
    for sample in synthetic_mobile + trajectory + productivity:
        retrieval_examples.setdefault(sample.ref_name, []).append(sample.prompt)
    route, selector, examples = _train_probe(
        model,
        tok,
        parent,
        tools,
        train_mobile,
        productivity_train,
        synthetic_mobile,
        trajectory,
        steps=args.steps,
        device=args.device,
        seed=2027,
        probe_init=args.probe_init,
        focus_tools=set(args.focus_tool),
        focus_repeat=args.focus_repeat,
    )
    pointer = None
    pointer_training: dict[str, int | float] = {
        "steps": 0,
        "examples": 0,
        "train_rows": 0,
        "held_out": {"rows": 0, "exact": 0, "exact_rate": 0.0},
    }
    if args.pointer_steps:
        pointer = _warm_pointer(parent, model.cfg.d_model).to(args.device)
        pointer_train_rows = (
            [sample for sample, _ in train_mobile]
            + synthetic_mobile
            + trajectory
            + productivity_train
        )
        pointer_eval_rows = [sample for sample, _ in held_mobile] + productivity_holdout
        train_info = _train_pointer(
            model,
            tok,
            pointer_train_rows,
            pointer,
            steps=args.pointer_steps,
            seed=2027,
            device=args.device,
        )
        pointer_training = {
            **train_info,
            "train_rows": len(pointer_train_rows),
            "held_out": _pointer_score(model, tok, pointer, pointer_eval_rows, args.device),
        }
    bound = BoundSelector(selector, tools, device=args.device, examples=examples)
    train_metrics = _score(
        model,
        tok,
        route,
        bound,
        train_mobile + [(sample, "productivity-train") for sample in productivity_train],
        args.device,
    )
    held_metrics = _score(model, tok, route, bound, held_mobile, args.device)
    held_productivity_metrics = _score(
        model,
        tok,
        route,
        bound,
        [(sample, "productivity-held") for sample in productivity_holdout],
        args.device,
    )

    child = dict(parent)
    child.update(
        {
            "stage": "sft_realistic_mobile_dispatch_pilot",
            "parent_checkpoint_sha256": _sha256(args.init)[1],
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "selector_proj": int(parent.get("selector_proj", 256)),
            "examples": examples,
            "retrieval_examples": retrieval_examples,
            "dispatch_tool_pool": [tool.name for tool in tools],
            "mobile_dispatch_training": {
                "rows": len(mobile),
                "episodes": episode_ids,
                "held_out_episodes": sorted(holdout_ids),
                "split_mode": split_mode,
                "external_eval_rows": len(held_mobile) if eval_rows else 0,
                "steps": args.steps,
                "probe_initialization": args.probe_init,
                "focus_tools": sorted(set(args.focus_tool)),
                "focus_repeat": args.focus_repeat,
                "pointer_training": pointer_training,
                "synthetic_mobile_rows": len(synthetic_mobile),
                "synthetic_mobile_repeats": 4,
                "state_conditioned_trajectory_rows": len(trajectory),
                "train": train_metrics,
                "held_out": held_metrics,
                "productivity_train": _score(
                    model,
                    tok,
                    route,
                    bound,
                    [(sample, "productivity-train") for sample in productivity_train],
                    args.device,
                ),
                "productivity_held_out": held_productivity_metrics,
            },
        }
    )
    if pointer is not None:
        child["ptr_head"] = pointer.state_dict()
        child["ptr_args"] = list(pointer.args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_realistic_mobile_dispatch_training_report",
        "parent": {"path": str(args.init), "sha256": _sha256(args.init)[1]},
        "child": {"path": str(args.output), "sha256": _sha256(args.output)[1]},
        "data": {
            "train": [
                {"path": str(path), "bytes": _sha256(path)[0], "sha256": _sha256(path)[1]}
                for path in args.data
            ],
            "eval": [
                {"path": str(path), "bytes": _sha256(path)[0], "sha256": _sha256(path)[1]}
                for path in args.eval_data
            ],
        },
        "tool_pool": [tool.name for tool in tools],
        "mobile_dispatch_training": child["mobile_dispatch_training"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
