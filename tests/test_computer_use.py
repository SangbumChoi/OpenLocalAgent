"""Computer-use + modern agentic tool family: generation, text-grounding, and episode shape.

The byte-level model is text-only (no vision), so every string arg of a computer-use call must be a
literal substring of the prompt (a semantic element description, never pixel coordinates). These
tests assert that discipline holds across a large sample with 0 grounding misses, that the new
tools are wired into CLASSES 1:1 with STANDARD_TOOLS, and that a GUI episode is well-formed.
"""

import json

from openlocalagent.agent.tool_head import CLASSES
from openlocalagent.agent.toolset import STANDARD_TOOLS
from openlocalagent.data.synth.agent_synth import Generator, episode_plan
from openlocalagent.data.schema import Role

COMPUTER_USE = ["screenshot", "click", "double_click", "type_text", "key_press", "scroll", "drag",
                "wait", "move_cursor", "open_app"]
MODERN_DEV = ["run_python", "edit_file", "apply_patch", "http_request", "sql_query", "list_dir",
              "find_files", "git_diff", "git_status", "install_package", "kill_process",
              "read_clipboard", "write_clipboard", "download_file", "unzip", "env_get", "make_dir",
              "list_processes", "docker_run"]
NEW_TOOLS = COMPUTER_USE + MODERN_DEV


def test_classes_match_toolset():
    names = [t.name for t in STANDARD_TOOLS]
    assert len(names) == len(set(names))                 # no duplicate tool names
    assert len(CLASSES) == len(STANDARD_TOOLS) + 1       # +1 for the "text"/abstain STOP class
    assert CLASSES[-1] == "text"                         # STOP/abstain stays LAST
    assert CLASSES[:-1] == names                         # 1:1, in order


def test_tool_count_is_fifty():
    assert len(STANDARD_TOOLS) == 50


def test_new_tools_present():
    names = {t.name for t in STANDARD_TOOLS}
    for n in NEW_TOOLS:
        assert n in names, n


def test_new_tools_generate_and_are_grounded():
    """0 grounding misses: every string arg is a literal substring; every int appears as digits."""
    misses = 0
    for split in ("train", "eval"):
        g = Generator(level=5, seed=11, split=split)
        for name in NEW_TOOLS:
            for _ in range(120):
                s = getattr(g, name)()
                assert s.ref_name == name
                assert s.kind == "tool"
                args = json.loads(s.ref_args) if s.ref_args else {}
                for v in args.values():
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, str):
                        if v not in s.prompt:
                            misses += 1
                    elif isinstance(v, int):
                        if str(v) not in s.prompt:
                            misses += 1
    assert misses == 0


def test_target_grounds_as_quoted_span_not_path():
    """`click(target)` declares format=quoted even though `target` is also a path-name hint —
    the value must ground as the quoted element description, not a (missing) file path."""
    from openlocalagent.agent.schema_decode import fill_tool
    byname = {t.name: t for t in STANDARD_TOOLS}
    got = fill_tool("Click 'the Submit button'.", byname["click"])
    assert got == {"target": "the Submit button"}


def test_constrained_decoder_grounds_new_tools():
    from openlocalagent.agent.constrained import candidates
    bodies = [b for b, _, _ in candidates("Click 'the Login button'.", STANDARD_TOOLS)]
    assert any('"target":"the Login button"' in b and "click" in b for b in bodies)
    bodies = [b for b, _, _ in candidates("Scroll down.", STANDARD_TOOLS)]
    assert any('"direction":"down"' in b for b in bodies)
    bodies = [b for b, _, _ in candidates("Wait 5 seconds.", STANDARD_TOOLS)]
    assert any('"seconds":5' in b for b in bodies)        # typed int, not "5"


def test_train_eval_slot_pools_disjoint():
    import openlocalagent.data.synth.agent_synth as A
    pairs = [
        (A.UI_TARGETS_TRAIN, A.UI_TARGETS_EVAL), (A.TYPED_TEXT_TRAIN, A.TYPED_TEXT_EVAL),
        (A.APPS_TRAIN, A.APPS_EVAL), (A.WAIT_SECONDS_TRAIN, A.WAIT_SECONDS_EVAL),
        (A.PYCODE_TRAIN, A.PYCODE_EVAL), (A.GLOBS_TRAIN, A.GLOBS_EVAL), (A.SQL_TRAIN, A.SQL_EVAL),
        (A.PACKAGES_TRAIN, A.PACKAGES_EVAL), (A.PROCESSES_TRAIN, A.PROCESSES_EVAL),
        (A.ENVVARS_TRAIN, A.ENVVARS_EVAL), (A.IMAGES_TRAIN, A.IMAGES_EVAL),
        (A.ARCHIVES_TRAIN, A.ARCHIVES_EVAL),
    ]
    for tr, ev in pairs:
        assert not (set(tr) & set(ev))


def test_computer_use_episode_well_formed():
    g = Generator(level=5, seed=3, split="train")
    for _ in range(40):
        ep = g.computer_use_episode()
        assert ep.meta["kind"] == "computer_use_episode"
        # alternating well-formed multi-turn: first turn user, has assistant tool-calls + a final text
        assert ep.messages[0].role == Role.user
        plan = episode_plan(ep)
        assert len(plan) >= 2                              # multi-step GUI flow
        # every tool-call in the episode targets a known tool name
        names = {t.name for t in STANDARD_TOOLS}
        for step in plan:
            assert step in names
        # the COPY args of each tool-call turn are grounded somewhere in the running context.
        # enum args (key_press.key, scroll.direction, http_request.method) ground semantically via
        # the enum extractor, not by copy, so they are exempt from the literal-substring check.
        spec = {t.name: t for t in STANDARD_TOOLS}
        ctx = ""
        for m in ep.messages:
            if m.role == Role.assistant and m.tool_calls:
                call = m.tool_calls[0]
                props = (spec[call.name].parameters or {}).get("properties", {})
                for k, v in call.arguments.items():
                    if isinstance(v, str) and "enum" not in props.get(k, {}):
                        assert v in ctx, (v, ctx)
            ctx += (m.content or "") + (m.tool_response or "")
            if m.tool_calls:
                ctx += json.dumps([c.arguments for c in m.tool_calls])


def test_gui_plan_episodes_registered():
    g = Generator(level=5, seed=1, split="train")
    builders = g._planner_builders()
    assert "plan_gui_login" in builders and "plan_gui_open_click" in builders
    ep = g._build_plan_episode("plan_gui_login")
    assert ep.meta["plan_len"] == 4
    assert ep.meta["plan"] == ["click", "type_text", "key_press", "screenshot"]
