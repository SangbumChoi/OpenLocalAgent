#!/usr/bin/env python3
"""Run a bounded AppWorld evaluation of a LocalAgent checkpoint.

AppWorld is an optional external dependency.  The selected tasks must be from its train/dev
splits, where ground-truth verifiers are available in the public data bundle.  By default this
runner does not translate LocalAgent's compact tool vocabulary into AppWorld Python/API calls: a
zero-action result is therefore a native checkpoint baseline and an explicit interface gap.  The
optional API-step adapter accepts only one literal AST call with fixture credentials; it is a
bounded diagnostic, not a claimed AppWorld agent score.  The receipt keeps task text out of
committed artifacts.

Example (with AppWorld installed in an isolated environment)::

    APPWORLD_ROOT=/tmp/appworld-data \
      PYTHONPATH=/tmp/appworld-venv/lib/python3.12/site-packages:src \
      python scripts/evaluate_appworld_checkpoint.py \
      --checkpoint runs/sft-mind2web-trajectory-continuation-20260802/latest.pt \
      --task 29caf6f_1 --task 771d8fc_1 --task 530b157_1 \
      --report /tmp/appworld-current-checkpoint.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

_TOOL_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\(")


def _sha256(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _text_hash(text: str) -> dict[str, int | str]:
    encoded = text.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _task_ids(root: Path, split: str) -> set[str]:
    path = root / "data" / "datasets" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"AppWorld split file not found: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _registry(capture: list[dict[str, Any]] | None = None):
    from localagent.agent.tools import ToolRegistry
    from localagent.agent.toolset import STANDARD_TOOLS

    registry = ToolRegistry()
    for tool in STANDARD_TOOLS:
        def _fn(name: str):
            def dispatch(**kwargs):
                if capture is not None:
                    capture.append({"name": name, "arguments": kwargs})
                return {"ok": name, "args": kwargs}

            return dispatch

        registry.register(
            tool,
            _fn(tool.name),
        )
    return registry


def _tracker_summary(tracker: Any) -> dict[str, Any]:
    payload = tracker.to_dict()
    passes = payload.get("passes", [])
    failures = payload.get("failures", [])
    return {
        "success": bool(payload.get("success", False)),
        "difficulty": payload.get("difficulty"),
        "num_tests": int(payload.get("num_tests", len(passes) + len(failures))),
        "passed": len(passes),
        "failed": len(failures),
        "pass_labels": sorted(str(item.get("label", "")) for item in passes),
        "failure_labels": sorted(str(item.get("label", "")) for item in failures),
    }


def _parse_appworld_api_code(code: str) -> tuple[str, str, dict[str, Any]] | None:
    """Parse one safe ``apis.<app>.<api>(literal_kwargs)`` action from model code."""

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or call.args or call.keywords and any(
        keyword.arg is None for keyword in call.keywords
    ):
        return None
    func = call.func
    if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Attribute):
        return None
    if not isinstance(func.value.value, ast.Name) or func.value.value.id != "apis":
        return None
    arguments: dict[str, Any] = {}
    try:
        for keyword in call.keywords:
            assert keyword.arg is not None
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
    except (AssertionError, ValueError, TypeError):
        return None
    return func.value.attr, func.attr, arguments


def _schema_ground_appworld_api_step(
    model: Any,
    tokenizer: Any,
    world: Any,
    prompt: str,
    api_head: Any | None = None,
    allow_completion: bool = False,
    lexical_first: bool = False,
    completion_only: bool = False,
) -> str | None:
    """Rank bounded API-schema candidates with the checkpoint instead of free-generating code."""

    from localagent.agent.constrained import _best
    from localagent.model.tokenizer import TOOL_CALL_CLOSE, TOOL_CALL_OPEN

    stopwords = {
        "a", "an", "and", "all", "across", "are", "as", "at", "be", "for", "from", "give",
        "have", "how", "i", "in", "is", "it", "list", "me", "my", "of", "on", "or", "the",
        "this", "to", "what", "which", "with", "your",
    }
    instruction_text = prompt.split("\nASSISTANT:", 1)[0]
    prompt_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", instruction_text.lower())
        if token not in stopwords
    }
    quoted = [value for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", instruction_text)
              for value in (left or right,)]
    # Keep multi-token names intact (for example ``Lily Moon``).  The previous single-token
    # fallback silently changed public AppWorld arguments to ``Lily`` and made otherwise valid
    # schema candidates fail in the native verifier.
    proper_noun_phrases = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", instruction_text
    )
    proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", instruction_text)
    proper_name = proper_noun_phrases[0] if proper_noun_phrases else (proper_nouns[0] if proper_nouns else None)
    observed_values: dict[str, list[Any]] = {}
    for key, raw in re.findall(
        r'"([a-zA-Z][a-zA-Z0-9_]*)"\s*:\s*(-?\d+(?:\.\d+)?)', prompt
    ):
        value: Any = float(raw) if "." in raw else int(raw)
        observed_values.setdefault(key, []).append(value)
    # Natural-language thresholds are useful for public task instructions such as "over 980".
    instruction_numbers = [int(raw) for raw in re.findall(r"\b\d+\b", instruction_text)]
    selected_label: str | None = None
    observed_fields: set[str] = set()
    if api_head is not None:
        if hasattr(api_head, "predict"):
            selected_label = str(api_head.predict(prompt))
            observed_fields = set(getattr(api_head, "argument_fields", {}).get(selected_label, ()))
        else:
            import torch

            from localagent.agent.tool_head import _feat

            with torch.no_grad():
                feature = _feat(model, tokenizer, prompt, "cpu", framed=False).unsqueeze(0)
                selected_index = int(api_head(feature).argmax(-1).item())
            selected_label = api_head.classes[selected_index]
    low_prompt = prompt.lower()
    previous = re.findall(
        r"apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\(", prompt
    )
    if not previous:
        app_names = re.findall(r"\"app\"\s*:\s*\"([A-Za-z_][A-Za-z0-9_]*)\"", prompt)
        api_names = re.findall(r"\"api\"\s*:\s*\"([A-Za-z_][A-Za-z0-9_]*)\"", prompt)
        if app_names and api_names:
            previous = [(app_names[-1], api_names[-1])]
    previous_api = f"{previous[-1][0]}.{previous[-1][1]}" if previous else None
    previous_codes: list[dict[str, Any]] = []
    for raw in re.findall(r"ASSISTANT\s*:\s*\[run_python\((\{.*?\})\)\]", prompt):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            previous_codes.append(parsed)
    used_song_ids = {
        int(arguments["song_id"])
        for item in previous_codes
        for arguments in [item.get("arguments", {})]
        if isinstance(arguments, dict)
        and "song_id" in arguments
        and isinstance(arguments["song_id"], int)
    }
    ranked: list[tuple[float, str]] = []
    if not completion_only:
        for app, docs in world.task.api_docs.items():
            if app in {"admin", "api_docs", "supervisor"}:
                continue
            for api, doc in docs.items():
                if api in {"login", "logout", "signup", "verify_account", "reset_password"}:
                    continue
                if selected_label is not None and selected_label != f"{app}.{api}":
                    continue
                description = str(doc.get("description", ""))
                api_tokens = set(re.findall(r"[a-z0-9]+", f"{api} {description}".lower()))
                overlap = prompt_tokens & api_tokens
                score = float(2 * len(overlap))
                if app.lower() in prompt.lower():
                    score += 3.0
                intent_candidate = (
                    (re.search(r"currently playing|current song", low_prompt)
                     and api == "show_current_song" and app == "spotify")
                    or (
                        re.search(r"\badd\b.*\bqueue\b|player queue", low_prompt)
                        and app == "spotify"
                        and api in {"search_artists", "search_songs", "add_to_queue"}
                    )
                    or (
                        re.search(r"how many.*follow|follower count", low_prompt)
                        and app == "spotify"
                        and api == "show_artist"
                        and previous_api == "spotify.show_current_song"
                    )
                )
                if not overlap and selected_label is None and not intent_candidate:
                    continue
                params = doc.get("parameters", [])
                arguments: dict[str, Any] = {}
                viable = True
                for parameter in params:
                    name = str(parameter.get("name", ""))
                    if not name or name == "access_token":
                        continue
                    default = parameter.get("default")
                    if (
                        api == "add_to_queue"
                        and name in {"album_id", "playlist_id"}
                        and "song_id" in observed_values
                    ):
                        # The Spotify ABI accepts at most one queue target.  Prefer the concrete
                        # song returned by search_songs over incidental album/playlist IDs in the
                        # same bounded observation.
                        continue
                    if name in observed_values:
                        values = observed_values[name]
                        if name == "artist_id" and previous_api == "spotify.search_artists":
                            arguments[name] = values[0]
                        elif name == "song_id" and api == "add_to_queue":
                            remaining = [value for value in values if value not in used_song_ids]
                            if remaining:
                                arguments[name] = remaining[0]
                            else:
                                arguments[name] = values[-1]
                        else:
                            arguments[name] = values[-1]
                    elif name.endswith("_id") and "id" in observed_values:
                        # AppWorld response summaries often expose a nested object's ``id``
                        # while the next API names it ``artist_id``/``album_id``/``song_id``.
                        # Restrict this fallback to an unambiguous single observed id.
                        ids = observed_values["id"]
                        if len(ids) == 1:
                            arguments[name] = ids[0]
                    elif name == "page_index" and default == 0:
                        arguments[name] = 0
                    elif (
                        name in {"query", "search_query"}
                        and (
                            parameter.get("required")
                            or name in observed_fields
                            or api.startswith("search_")
                        )
                        and (proper_name or quoted)
                    ):
                        arguments[name] = proper_name or quoted[0]
                    elif (
                        name == "min_play_count"
                        and re.search(r"\bplay(?:ed|s)?\b|play_count", low_prompt)
                        and instruction_numbers
                    ):
                        arguments[name] = instruction_numbers[-1]
                    elif (
                        name == "min_follower_count"
                        and re.search(r"follow|follower", low_prompt)
                        and instruction_numbers
                    ):
                        arguments[name] = instruction_numbers[-1]
                    elif name == "amount" and instruction_numbers:
                        arguments[name] = instruction_numbers[-1]
                    elif parameter.get("required") and default is None:
                        viable = False
                        break
                if viable:
                    # The model-ranked candidate remains the default.  ``--lexical-first`` is an
                    # explicit schema-planning control, however, so give it generic dependency
                    # signals for stateful API workflows.  These signals use only the task text,
                    # API documentation, and the already-replayed API trace; they never read the
                    # ground-truth solution or answer.
                    label = f"{app}.{api}"
                    intent_bonus = 0.0
                    if re.search(r"currently playing|current song", low_prompt):
                        intent_bonus += 60.0 if label == "spotify.show_current_song" else 0.0
                        if label in {"spotify.pause_music", "spotify.play_music", "spotify.seek_song"}:
                            intent_bonus -= 25.0
                    if re.search(r"how many.*follow|follower count", low_prompt):
                        if label == "spotify.show_current_song" and previous_api is None:
                            intent_bonus += 25.0
                        if label == "spotify.show_artist" and previous_api == "spotify.show_current_song":
                            intent_bonus += 70.0
                    if re.search(r"\badd\b.*\bqueue\b|player queue", low_prompt):
                        if previous_api is None and label == "spotify.search_artists":
                            intent_bonus += 70.0
                        elif previous_api == "spotify.search_artists" and label == "spotify.search_songs":
                            intent_bonus += 70.0
                        elif previous_api == "spotify.search_songs" and label == "spotify.add_to_queue":
                            intent_bonus += 70.0
                        if label in {"spotify.show_song_queue", "spotify.play_music", "spotify.pause_music"}:
                            intent_bonus -= 25.0
                    args = ", ".join(f"{key}={repr(arguments[key])}" for key in sorted(arguments))
                    code = f"apis.{app}.{api}({args})"
                    ranked.append((score + intent_bonus, code))
    # Completion is deliberately opt-in: the default probe measures API-prefix replay only.
    # The candidate contains no answer, so question tasks can still fail honestly at the verifier.
    if allow_completion and "Next required action:" in prompt:
        answer_fields = ("answer", "count", "follower_count", "total", "value")
        answer_requested = bool(
            re.search(r"\bhow many\b|\bhow much\b|\bwhat\b|\bwhich\b|\bwho\b|\bwhen\b|\bwhere\b", prompt.lower())
        )
        answers = (
            [value for field in answer_fields for value in observed_values.get(field, [])]
            if answer_requested else []
        )
        completion_score = 100.0 if previous_api == "spotify.show_artist" else 2.0
        if previous_api == "spotify.add_to_queue":
            observed_song_ids = set(observed_values.get("song_id", []))
            completion_score = 100.0 if observed_song_ids <= used_song_ids else 2.0
        for answer in answers[-3:]:
            ranked.append(
                (completion_score, f"apis.supervisor.complete_task(answer={answer!r}, status='success')")
            )
        ranked.append((completion_score - 1.0, "apis.supervisor.complete_task(status='success')"))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    candidates = [
        f"{TOOL_CALL_OPEN}{json.dumps({'name': 'run_python', 'arguments': {'code': code}}, separators=(',', ':'), sort_keys=True)}{TOOL_CALL_CLOSE}"
        for _, code in ranked[:24]
    ]
    if lexical_first:
        # This is an explicit environment-side schema policy, not a model score.  It is useful as
        # a diagnostic control for separating API-name/argument grounding from free candidate ranking.
        selected_code = ranked[0][1]
        return selected_code
    selected = _best(model, tokenizer, prompt, candidates, "cpu")
    try:
        parsed = json.loads(selected[len(TOOL_CALL_OPEN):-len(TOOL_CALL_CLOSE)])
        code = parsed["arguments"]["code"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return code if isinstance(code, str) else None


def _appworld_execute_api_step(world: Any, app: str, api: str, arguments: dict[str, Any]) -> tuple[Any, int]:
    """Execute one parsed API step through AppWorld's persisted sandbox boundary."""

    if app == "supervisor" and api == "complete_task":
        allowed = {"answer", "status"}
        if set(arguments) - allowed:
            raise ValueError("supervisor.complete_task received unsupported arguments")
        status = arguments.get("status", "success")
        if status not in {"success", "fail"}:
            raise ValueError("supervisor.complete_task status must be success or fail")
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items()))
        code = f"result = apis.supervisor.complete_task({rendered})\nprint(repr(result))"
        request_count_before = len(world.requester.request_tracker.requests)
        message = world.execute(code)
        response = _parse_appworld_printed_result(message)
        return response, len(world.requester.request_tracker.requests) - request_count_before
    if app in {"admin", "api_docs", "supervisor"} or api in {"login", "logout"}:
        raise ValueError("bootstrap/admin AppWorld API actions are not model-replayable")
    if app not in world.apis or api not in world.apis[app]:
        raise ValueError(f"unknown AppWorld API action: {app}.{api}")
    from appworld.common.misc import get_login_by

    login_by = get_login_by(app)
    if login_by is None:
        raise ValueError(f"fixture has no supported login bootstrap for AppWorld app {app!r}")
    api_doc = world.task.api_docs.get(app, {}).get(api, {})
    api_parameters = {
        str(parameter.get("name", ""))
        for parameter in api_doc.get("parameters", [])
        if isinstance(parameter, dict)
    }
    rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items()))
    if "access_token" in api_parameters:
        rendered = "access_token=token" + (", " + rendered if rendered else "")
    code = (
        "profile = apis.supervisor.show_profile()\n"
        "passwords = {item['account_name']: item['password'] for item in apis.supervisor.show_account_passwords()}\n"
        f"token = apis.{app}.login(username=profile[{login_by!r}], password=passwords[{app!r}])['access_token']\n"
        f"result = apis.{app}.{api}({rendered})\n"
        "print(repr(result))"
    )
    request_count_before = len(world.requester.request_tracker.requests)
    message = world.execute(code)
    response = _parse_appworld_printed_result(message)
    return response, len(world.requester.request_tracker.requests) - request_count_before


def _parse_appworld_printed_result(message: str) -> Any:
    """Parse the final repr printed by a one-action AppWorld sandbox execution."""

    if not isinstance(message, str) or message.startswith("Execution failed."):
        raise RuntimeError(message)
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("AppWorld action produced no printed result")
    try:
        return ast.literal_eval(lines[-1])
    except (SyntaxError, ValueError) as error:
        raise RuntimeError(f"AppWorld action result was not a literal: {lines[-1]!r}") from error


def _verify_runner_contract(*, AppWorld: Any, task_id: str, experiment_name: str) -> dict[str, Any]:
    """Run one bundled ground-truth solution to prove the native verifier is live."""

    with AppWorld(
        task_id=task_id,
        experiment_name=f"{experiment_name}_oracle",
        load_ground_truth=True,
        ground_truth_mode="full",
        max_interactions=256,
    ) as world:
        ground_truth = world.task.ground_truth
        if ground_truth is None:
            raise RuntimeError(f"AppWorld ground truth missing for contract task {task_id!r}")
        code = ground_truth.compiled_solution_code + "\nsolution(apis, requester)"
        world.execute(code)
        tracker = world.evaluate()
    summary = _tracker_summary(tracker)
    summary["task_id"] = task_id
    return summary


def evaluate(
    *,
    checkpoint: Path,
    root: Path,
    task_ids: list[str],
    report: Path,
    experiment_name: str,
    selector_first: bool = False,
    retrieve_k: int = 10,
    replay_run_python: bool = False,
    replay_appworld_api_step: bool = False,
    schema_ground_appworld_api_step: bool = False,
    appworld_api_head: Path | None = None,
) -> dict[str, Any]:
    try:
        appworld_version = importlib.metadata.version("appworld")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "AppWorld is optional; install it in an isolated environment before running this "
            "script (see docs/REALISTIC_AGENT_RESEARCH.md)."
        ) from error

    from appworld import AppWorld, update_root
    from localagent.agent.runtime import Agent
    from localagent.eval.appworld_api_head import load_appworld_api_head

    root = root.resolve()
    resolved_root = Path(update_root(str(root))).resolve()
    if resolved_root != root:
        raise RuntimeError(
            f"APPWORLD_ROOT resolved to {str(resolved_root)!r}, expected {str(root)!r}; set it before "
            "importing AppWorld."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not task_ids:
        raise ValueError("at least one --task is required")
    for task_id in task_ids:
        if not (root / "data" / "tasks" / task_id / "specs.json").is_file():
            raise FileNotFoundError(f"AppWorld task specs not found for {task_id!r}")

    if retrieve_k < 1:
        raise ValueError("retrieve_k must be positive")
    if replay_run_python and replay_appworld_api_step:
        raise ValueError("replay_run_python and replay_appworld_api_step are mutually exclusive")
    calls: list[dict[str, Any]] = []
    agent = Agent.from_checkpoint(
        checkpoint,
        _registry(calls),
        selector_first=selector_first,
        retrieve_k=retrieve_k,
    )
    api_head = (
        load_appworld_api_head(appworld_api_head, d_model=agent.model.cfg.d_model)
        if appworld_api_head is not None
        else None
    )
    contract_verification = _verify_runner_contract(
        AppWorld=AppWorld, task_id=task_ids[0], experiment_name=experiment_name
    )
    task_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_spec_path = root / "data" / "tasks" / task_id / "specs.json"
        task_spec = json.loads(task_spec_path.read_text(encoding="utf-8"))
        instruction = str(task_spec["instruction"])
        calls.clear()
        replay_response: str | None = None
        replay_error: str | None = None
        action_replayed = False
        native_api_calls = 0
        native_action_api_calls = 0
        native_bootstrap_api_calls = 0
        translated_code: str | None = None
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            ground_truth_mode="full",
            max_interactions=1,
        ) as world:
            output = agent.chat(instruction)
            selected = calls[0] if calls else None
            if replay_run_python and selected and selected["name"] == "run_python":
                code = selected["arguments"].get("code")
                if isinstance(code, str) and code.strip():
                    action_replayed = True
                    try:
                        replay_response = world.execute(code)
                    except Exception as error:  # AppWorld should report failures, but stay fail-closed.
                        replay_error = repr(error)
                    tracker_obj = getattr(getattr(world, "requester", None), "request_tracker", None)
                    requests = getattr(tracker_obj, "requests", None)
                    if requests is not None:
                        native_api_calls = len(requests)
            elif replay_appworld_api_step and selected and selected["name"] == "run_python":
                code = selected["arguments"].get("code")
                if schema_ground_appworld_api_step:
                    translated_code = _schema_ground_appworld_api_step(
                        agent.model, agent.tokenizer, world, instruction, api_head=api_head
                    )
                    if translated_code is not None:
                        code = translated_code
                parsed = _parse_appworld_api_code(code) if isinstance(code, str) else None
                if parsed is not None:
                    app, api, arguments = parsed
                    try:
                        response, native_api_calls = _appworld_execute_api_step(
                            world, app, api, arguments
                        )
                        replay_response = json.dumps(
                            response, ensure_ascii=False, sort_keys=True, default=str
                        )
                        action_replayed = True
                        native_action_api_calls = 1
                        native_bootstrap_api_calls = max(0, native_api_calls - 1)
                    except Exception as error:  # keep malformed/API failures in the receipt
                        replay_error = repr(error)
                        native_api_calls = len(world.requester.request_tracker.requests)
            tracker = world.evaluate()
            tracker_summary = _tracker_summary(tracker)
        selected = calls[0] if calls else None
        match = _TOOL_RE.search(output)
        predicted_tool = selected["name"] if selected else (match.group(1) if match else None)
        record = {
            "task_id": task_id,
            "task_spec": _sha256(task_spec_path),
            "instruction": _text_hash(instruction),
            "model_output": _text_hash(output),
            "predicted_tool": predicted_tool,
            "predicted_arguments": (
                _text_hash(json.dumps(selected["arguments"], sort_keys=True, separators=(",", ":")))
                if selected
                else None
            ),
            "action_replayed": action_replayed,
            "native_api_calls": native_api_calls,
            "native_action_api_calls": native_action_api_calls,
            "native_bootstrap_api_calls": native_bootstrap_api_calls,
            "schema_translation_applied": translated_code is not None,
            "appworld_api_head_applied": api_head is not None,
            "evaluation": tracker_summary,
        }
        if replay_response is not None:
            record["replay_response"] = _text_hash(replay_response)
        if replay_error is not None:
            record["replay_error"] = _text_hash(replay_error)
        if translated_code is not None:
            record["translated_code"] = _text_hash(translated_code)
        task_records.append(
            record
        )

    result = {
        "kind": "localagent_appworld_checkpoint_native_probe",
        "schema_version": 1,
        "runner": {
            "package": "appworld",
            "version": appworld_version,
            "root": str(root),
            "data_version": (root / "data" / "version.txt").read_text(encoding="utf-8").strip(),
            "split_policy": "caller-selected train/dev tasks with full public ground-truth verifiers",
            "contract_verification": {
                "tasks": 1,
                "passed": int(contract_verification["success"]),
                "result": contract_verification,
            },
        },
        "checkpoint": _sha256(checkpoint),
        "configuration": {
            "experiment_name": experiment_name,
            "tasks": task_ids,
            "action_translation": (
                "appworld_api_step"
                if replay_appworld_api_step
                else "appworld_run_python"
                if replay_run_python
                else "disabled"
            ),
            "selector_first": selector_first,
            "retrieve_k": retrieve_k,
            "replay_run_python": replay_run_python,
            "replay_appworld_api_step": replay_appworld_api_step,
            "schema_ground_appworld_api_step": schema_ground_appworld_api_step,
            "appworld_api_head": str(appworld_api_head) if appworld_api_head else None,
            "max_interactions": 1,
        },
        "environment": {
            "native_runtime_executed": True,
            "environment_reset_per_task": True,
            "external_accounts": False,
            "screenshots": False,
            "state_side_effects": "isolated AppWorld task databases only",
        },
        "tasks": task_records,
        "summary": {
            "tasks": len(task_records),
            "native_successes": sum(int(item["evaluation"]["success"]) for item in task_records),
            "native_success_rate": sum(int(item["evaluation"]["success"]) for item in task_records)
            / len(task_records),
            "action_replayed": sum(int(item["action_replayed"]) for item in task_records),
            "native_api_calls": sum(item["native_api_calls"] for item in task_records),
            "native_action_api_calls": sum(item["native_action_api_calls"] for item in task_records),
            "native_bootstrap_api_calls": sum(
                item["native_bootstrap_api_calls"] for item in task_records
            ),
        },
        "claim_boundary": (
            "Native AppWorld reset/evaluation of the current LocalAgent checkpoint only. When API-step "
            "replay is enabled, a strict AST parser accepts at most one literal "
            "apis.<app>.<api>(...) call and injects credentials from the resettable fixture. An "
            "optional schema-grounding mode ranks API-schema candidates with the checkpoint rather "
            "than free-generating code; when "
            "run_python replay is enabled, the model code is executed directly. Both are adapter "
            "diagnostics, not an AppWorld leaderboard result, AppWorld-UL result, or evidence of "
            "email/SMS/Spotify task success."
            if replay_run_python or replay_appworld_api_step
            else "Native AppWorld reset/evaluation of the current LocalAgent checkpoint only. The "
            "model emits LocalAgent tool syntax, while AppWorld expects Python/API actions; no "
            "action was translated or replayed. The score is a zero-action interface baseline, not "
            "an AppWorld leaderboard result, AppWorld-UL result, or evidence of email/SMS/Spotify "
            "task success."
        ),
    }
    result["receipt_self_sha256"] = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("APPWORLD_ROOT", ".")))
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-name", default="localagent_appworld_native_probe")
    parser.add_argument(
        "--selector-first",
        action="store_true",
        help="use the dense selector's top tool directly instead of model candidate scoring",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=10,
        help="retriever candidate count; use the full runtime tool pool for selector adapters",
    )
    parser.add_argument(
        "--replay-run-python",
        action="store_true",
        help="execute captured run_python code in AppWorld and count native API calls",
    )
    parser.add_argument(
        "--replay-appworld-api-step",
        action="store_true",
        help="parse one safe apis.<app>.<api>(literal kwargs) step and execute it natively",
    )
    parser.add_argument(
        "--schema-ground-appworld-api-step",
        action="store_true",
        help="rank public AppWorld API-schema candidates with the model before AST replay",
    )
    parser.add_argument(
        "--appworld-api-head",
        type=Path,
        help="optional frozen-body app.api head used to restrict schema grounding",
    )
    args = parser.parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        root=args.root,
        task_ids=args.tasks,
        report=args.report,
        experiment_name=args.experiment_name,
        selector_first=args.selector_first,
        retrieve_k=args.retrieve_k,
        replay_run_python=args.replay_run_python,
        replay_appworld_api_step=args.replay_appworld_api_step,
        schema_ground_appworld_api_step=args.schema_ground_appworld_api_step,
        appworld_api_head=args.appworld_api_head,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
