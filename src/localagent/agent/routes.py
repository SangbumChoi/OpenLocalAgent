"""Route taxonomy — the small, *stable* closed set the selection head should classify.

The previous design put a 51-way softmax (`tool_head.CLASSES`) over every concrete tool. That is the
wrong abstraction: a fixed N-way classifier can't accept a tool it didn't see at train time and has
to be reshaped+retrained whenever the tool pool changes (the 22->51 jump already forced a *fresh*
head). It also doesn't transfer to the function-calling / MCP setting where the available tools
differ per request.

The fix: the head classifies one of a few **high-level routes** (modalities) — a small set that is
stable across tool pools — and the *specific* `tool(args)` is then **generated as text** by the LM
(see `eval.harness.evaluate_routed`). Adding/removing a concrete tool no longer reshapes the head;
only adding a whole new modality does, which is rare.

ROUTES (5):
  web_search    — search / browse / fetch information over the network
  computer_use  — GUI control (screenshot, click, type, scroll, ...)
  code          — run code / commands, file & dev ops, compute
  app_action    — side-effecting integrations (email, calendar, slack, reminders, ...)
  text          — answer directly in natural language (no tool / abstain)
"""

from __future__ import annotations

ROUTES = ["web_search", "computer_use", "code", "app_action", "text"]
ROUTE_INDEX = {r: i for i, r in enumerate(ROUTES)}

# Concrete tool -> route. Covers every name in tool_head.CLASSES (a test asserts full coverage so
# this never silently drifts when the tool pool grows).
_TOOL_ROUTE: dict[str, str] = {
    # --- web_search: information retrieval over the network -------------------------------------
    "web_search": "web_search", "open_url": "web_search", "http_request": "web_search",
    "download_file": "web_search", "get_news": "web_search", "get_weather": "web_search",
    "define": "web_search",
    # --- computer_use: GUI control -------------------------------------------------------------
    "screenshot": "computer_use", "click": "computer_use", "double_click": "computer_use",
    "type_text": "computer_use", "key_press": "computer_use", "scroll": "computer_use",
    "drag": "computer_use", "wait": "computer_use", "move_cursor": "computer_use",
    "open_app": "computer_use", "read_clipboard": "computer_use", "write_clipboard": "computer_use",
    "web_click": "computer_use", "web_type": "computer_use", "web_select": "computer_use",
    "mobile_click": "computer_use", "mobile_long_press": "computer_use",
    "mobile_scroll": "computer_use", "mobile_swipe": "computer_use",
    "mobile_open_app": "computer_use", "mobile_input_text": "computer_use",
    "mobile_navigate_home": "computer_use", "mobile_navigate_back": "computer_use",
    "mobile_press_enter": "computer_use", "mobile_wait": "computer_use",
    "mobile_submit_answer": "computer_use",
    # --- code: run code/commands, files, dev ops, compute --------------------------------------
    "calculator": "code", "planner": "code", "run_python": "code", "run_command": "code",
    "read_file": "code", "write_file": "code", "edit_file": "code", "apply_patch": "code",
    "grep_search": "code", "git_commit": "code", "git_diff": "code", "git_status": "code",
    "run_tests": "code", "sql_query": "code", "list_dir": "code", "find_files": "code",
    "install_package": "code", "kill_process": "code", "make_dir": "code", "unzip": "code",
    "env_get": "code", "list_processes": "code", "docker_run": "code",
    # --- app_action: side-effecting integrations -----------------------------------------------
    "send_email": "app_action", "calendar_event": "app_action", "slack_send": "app_action",
    "jira_issue": "app_action", "notion_write": "app_action", "set_reminder": "app_action",
    "set_timer": "app_action", "play_music": "app_action",
    "email_send": "app_action", "notion_create_page": "app_action",
    # --- text: direct answer (no tool) ---------------------------------------------------------
    "text": "text",
}


def route_of(tool_name: str) -> str:
    """Map a concrete tool name to its route. Unknown names fall back to ``text`` (treated as a
    direct answer rather than crashing — keeps the head usable on out-of-pool tools)."""
    return _TOOL_ROUTE.get(tool_name, "text")


def route_of_sample(sample) -> str:
    """Route label for a synth ``Sample``: ``text`` for text turns, else the route of the (first)
    called tool. Mirrors ``tool_head.label_of`` but at route granularity."""
    if getattr(sample, "kind", None) != "tool":
        return "text"
    return route_of(sample.ref_name)


def RouteHead(d_model: int):
    """A 5-way modality head — the same tiny linear probe as ToolHead, but over ROUTES instead of
    the 51 concrete tools. Returned as a ToolHead so it plugs into existing `head(feat).argmax`
    call sites; `.classes` is ROUTES."""
    from localagent.agent.tool_head import ToolHead
    return ToolHead(d_model, classes=ROUTES)


def train_route_head(model, samples, tok, *, steps=300, batch_size=64, lr=5e-3, device="cpu",
                     log=lambda *a: None):
    """Train the route head as a frozen-feature linear probe (cheap; ~seconds). Mirrors
    `tool_head.train_tool_head` but labels are ROUTES via `route_of_sample`."""
    import random

    import torch
    import torch.nn.functional as F

    from localagent.agent.tool_head import _feat

    model.eval()
    head = RouteHead(model.cfg.d_model).to(device)
    with torch.no_grad():   # frozen-feature probe: cache detached features (no autograd graph/leak)
        feats = torch.stack(
            [
                _feat(model, tok, s.prompt, device, framed=bool(getattr(s, "framed", False)))
                for s in samples
            ]
        )
    labels = torch.tensor([ROUTE_INDEX[route_of_sample(s)] for s in samples], device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    rng = random.Random(0)
    n = len(samples)
    for step in range(steps):
        idx = torch.tensor([rng.randrange(n) for _ in range(batch_size)], device=device)
        loss = F.cross_entropy(head(feats[idx]), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            acc = (head(feats).argmax(-1) == labels).float().mean().item()
            log(f"  [route-head] step {step}/{steps} loss {loss.item():.3f} train-acc {acc:.3f}")
    return head
