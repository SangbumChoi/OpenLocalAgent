"""Learned planner rollout (planner -> action decomposition, stage 2).

The stage-1 ``ToolCaller.plan`` decomposed a multi-step request with a *regex* connective-splitter.
This module replaces that heuristic with a **learned, autoregressive rollout** that reuses the
EXISTING dual head (``tool_head`` for selection, ``ptr_head`` for argument grounding) — no new model
parameters. From the user query alone the planner iteratively (a) picks the next tool with the tool
head, (b) STOPs when the head emits its no-tool ``"text"`` class (or ``max_steps`` is hit), else (c)
grounds the chosen tool's args with the same grounded action decoder the single-call path uses, and
(d) appends a simulated tool response so the next step can ground a pointer arg out of it.

STOP signal: the tool head's ``"text"`` class is the learnable stop. The 0-step plan episodes
(``plan_no_tool_thanks`` / ``plan_no_tool_greet``) and the single-turn text/no_tool samples teach
the head to emit ``"text"`` when no (further) tool is warranted; the 1-step plan episodes teach it
to stop after exactly one tool. (The head is trained on the *running rendered context* ending at the
assistant marker — the very thing this rollout feeds it — so the stop class transfers in-distribution.)
"""

from __future__ import annotations

from localagent.data.schema import ToolCall, ToolSpec
from localagent.model.tokenizer import ASSISTANT, TOOL, TOOL_RESPONSE_CLOSE, TOOL_RESPONSE_OPEN, USER

# Tools whose synth tool-response "returns" a path/url that a downstream step grounds a pointer arg
# from. We mirror the synth response templates so an in-distribution value is present in the running
# context for the next step's pointer head to copy. Anything else gets a compact generic stub.
_PATH_TOOLS = {"grep_search", "run_command"}
_URL_TOOLS = {"web_search", "get_news"}


def _sim_response(call: ToolCall) -> str:
    """A simulated tool response for `call`, reusing the synth's shape where a downstream step
    needs to ground a pointer arg from it (a path after grep/run_command, a url after search/news).
    Otherwise a compact generic stub — enough to continue the rollout in-distribution."""
    name = call.name
    if name in _PATH_TOOLS:
        # the synth "returns" a path like "src/main.py:12:    TODO appears here"
        pat = call.arguments.get("pattern") or call.arguments.get("command") or ""
        return f"src/main.py:12:    {pat} appears here".rstrip()
    if name in _URL_TOOLS:
        q = call.arguments.get("query") or call.arguments.get("topic") or ""
        return f"1. example.com — overview of {q}".rstrip()
    if name == "run_tests":
        return "All tests passed."
    if name == "read_file":
        return "<file contents>"
    if name == "write_file":
        return "written."
    if name == "git_commit":
        return "Committed abc123."
    return "done."


def plan_rollout(model, tok, query: str, tools: list[ToolSpec], *, tool_head, ptr_head,
                 max_steps: int = 4, device: str = "cpu") -> list[ToolCall]:
    """Learned planner rollout: from `query` alone, autoregressively build an ordered plan and
    ground each step into a concrete ``ToolCall``, reusing the trained `tool_head`/`ptr_head` (no
    new parameters). Returns the executed plan as an ordered list of grounded ``ToolCall``s.

    At each step the running context is rendered exactly as ``multi_turn_eval`` /
    ``history_text`` render it (``<|user|>query`` then, per completed step,
    ``<|assistant|><tool_call>...</tool_call><|tool|><tool_response>...</tool_response>``, ending at
    ``<|assistant|>``). The tool head is applied at the final position; if it picks the no-tool
    ``"text"`` class — or `max_steps` is reached — the rollout STOPs. Otherwise the chosen tool's
    args are grounded by the same grounded action decoder used for single calls (``ptr_head`` copy
    spans + schema/heuristic extractors), the resulting call is appended, and a simulated tool
    response is appended to the context so the next step can ground a pointer arg from it."""
    from localagent.agent.constrained import _ctx_feats, _tool_bodies, _best
    from localagent.agent.tool_head import CLASSES
    from localagent.data.render import _canon
    from localagent.model.tokenizer import TOOL_CALL_CLOSE, TOOL_CALL_OPEN

    from localagent.agent.parser import extract_tool_calls

    by_name = {t.name: t for t in tools}
    # `history` is the marked-up running context (matches data.render.history_text); `plain` is the
    # accumulated user query + simulated tool-response text the heuristic extractors read (so a
    # path/url "returned" by an earlier step is reachable to non-pointer args too).
    history = USER + query
    plain = query
    plan: list[ToolCall] = []
    for _ in range(max_steps):
        ctx = history + ASSISTANT
        feats, ids = _ctx_feats(model, tok, ctx, device)
        picked = CLASSES[int(tool_head(feats[-1]).argmax(-1))]
        if picked == "text" or picked not in by_name:
            break                    # STOP: no (further) tool warranted
        tool = by_name[picked]
        # ground the chosen tool's args over the WHOLE running context: the pointer head reads it
        # via `feats`/`ids`; the heuristic string/path/url extractors read `plain`. Same grounded
        # action decoder the single-call path (grounded_decode) uses — no new parameters.
        ptr = (ptr_head, feats, ids, tok) if ptr_head is not None else None
        bodies = _tool_bodies(plain, tool, ptr)
        if not bodies:
            break
        best = _best(model, tok, ctx, bodies, device)
        calls = extract_tool_calls(best)
        if not calls:
            break
        call = calls[0]
        plan.append(call)
        body = TOOL_CALL_OPEN + _canon(call.name, call.arguments) + TOOL_CALL_CLOSE
        sim = _sim_response(call)
        history = history + ASSISTANT + body + TOOL + TOOL_RESPONSE_OPEN + sim + TOOL_RESPONSE_CLOSE
        plain = plain + " " + sim    # make the returned path/url visible to heuristic extractors
    return plan
