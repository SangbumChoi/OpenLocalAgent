from openlocalagent.data.synth.agent_synth import PATHS_EVAL, QUERIES_EVAL, Generator
from openlocalagent.data.render import IGNORE, history_text, render_conversation
from openlocalagent.data.schema import Role
from openlocalagent.model.tokenizer import load_tokenizer
from openlocalagent.agent.pointer_head import PointerHead, gold_span


def test_episode_is_multiturn_with_tool_response():
    conv = Generator(3, 0, "train").coding_episode()
    roles = [m.role for m in conv.messages]
    assert Role.user in roles and Role.tool in roles
    assert sum(m.role == Role.assistant and bool(m.tool_calls) for m in conv.messages) >= 1


def test_all_coding_types_are_producible():
    g = Generator(5, 0, "train")
    seen = set()
    for _ in range(400):
        seen.add(g.coding_episode().meta["type"])
    assert seen == set(Generator._CODING_TYPES)
    # the new ones in particular
    for t in ("refactor_rename", "review_pr", "dependency_bump", "trace_import", "lint_and_fix"):
        assert t in seen


def test_productivity_and_planner_episodes_exist():
    g = Generator(5, 0, "train")
    prod = {g.productivity_episode().meta["type"] for _ in range(300)}
    # planner_episode() yields the multi-step (>=1 tool call) plan types; 0-step "don't over-plan"
    # cases come from plan_episode()/plan_episodes(). plan_episodes() covers ALL plan types.
    plan = {g.planner_episode().meta["type"] for _ in range(400)}
    all_plan = {g.plan_episode().meta["type"] for _ in range(1500)}
    multi_types = {t for t, (_a, ln) in g._PLAN_BUILDERS.items() if ln >= 1}
    assert prod == set(Generator._PRODUCTIVITY_TYPES)
    assert plan == multi_types
    assert all_plan == set(Generator._PLANNER_TYPES)
    # productivity episodes are not coding episodes
    assert g.productivity_episode().meta["kind"] == "productivity_episode"
    assert g.planner_episode().meta["kind"] == "planner_episode"


def test_planner_episode_starts_with_canonical_text_plan():
    g = Generator(5, 0, "train")
    for _ in range(50):
        conv = g.planner_episode()
        # first assistant turn is a text plan, before any tool call
        first_asst = next(m for m in conv.messages if m.role == Role.assistant)
        assert not first_asst.tool_calls
        assert first_asst.content.startswith("Plan: 1)")
        # and at least one tool-call turn follows
        assert any(m.role == Role.assistant and m.tool_calls for m in conv.messages)


def test_episodes_mix_spans_all_kinds():
    eps = Generator(5, 1, "train").episodes(400)
    kinds = {e.meta["kind"] for e in eps}
    assert kinds == {
        "coding_episode",
        "productivity_episode",
        "planner_episode",
        "computer_use_episode",
    }


def test_episodes_mix_false_is_coding_only():
    eps = Generator(5, 1, "train").episodes(60, mix=False)
    assert all(e.meta["kind"] == "coding_episode" for e in eps)


def test_followup_arg_grounded_from_tool_response():
    """At least one episode type must put a follow-up tool-call arg ONLY in a prior tool response
    (the pointer-head case): the value is absent from every user turn but present in a tool turn,
    and is then used as a later tool-call argument."""
    g = Generator(5, 7, "train")
    found = False
    for _ in range(400):
        conv = g.coding_episode()
        if conv.meta["type"] != "grep_read":
            continue
        users = " ".join(m.content for m in conv.messages if m.role == Role.user)
        tools = " ".join(m.tool_response or "" for m in conv.messages if m.role == Role.tool)
        # the read_file path argument
        read = next(
            m
            for m in conv.messages
            if m.role == Role.assistant and m.tool_calls and m.tool_calls[0].name == "read_file"
        )
        path = read.tool_calls[0].arguments["path"]
        if path not in users and path in tools:
            found = True
            break
    assert found


def test_render_conversation_masks_nonassistant():
    tok = load_tokenizer("byte")
    conv = Generator(3, 1, "train").coding_episode()
    ids, labels = render_conversation(conv, tok)
    assert len(ids) == len(labels)
    assert any(lab != IGNORE for lab in labels)  # assistant turns are learned
    assert any(lab == IGNORE for lab in labels)  # user/tool turns are masked


def test_every_episode_renders_with_learned_and_masked_labels():
    tok = load_tokenizer("byte")
    eps = Generator(5, 2, "train").episodes(120)
    for conv in eps:
        ids, labels = render_conversation(conv, tok)
        assert len(ids) == len(labels) and ids
        assert any(lab != IGNORE for lab in labels)  # assistant body learned
        assert any(lab == IGNORE for lab in labels)  # user/tool masked
        history_text(conv.messages)  # renders cleanly too


def test_eval_split_episodes_use_eval_pools_only():
    g = Generator(5, 3, "eval")
    paths, queries = set(PATHS_EVAL), set(QUERIES_EVAL)
    train_paths_seen = False
    for _ in range(200):
        for conv in (g.coding_episode(), g.productivity_episode(), g.planner_episode()):
            for m in conv.messages:
                for c in m.tool_calls:
                    if "path" in c.arguments:
                        assert c.arguments["path"] in paths
                        train_paths_seen = True
                    if "query" in c.arguments:
                        assert c.arguments["query"] in queries
    assert train_paths_seen  # we actually exercised the path check


def test_every_episode_family_uses_disjoint_train_eval_user_phrasings():
    train = Generator(5, 101, "train")
    evaluation = Generator(5, 202, "eval")
    family_builders = [
        (train._coding_builders(), evaluation._coding_builders()),
        (train._productivity_builders(), evaluation._productivity_builders()),
        (train._computer_use_builders(), evaluation._computer_use_builders()),
        (train._planner_builders(), evaluation._planner_builders()),
    ]
    train_prompts: set[str] = set()
    eval_prompts: set[str] = set()
    for train_builders, eval_builders in family_builders:
        assert train_builders.keys() == eval_builders.keys()
        for name in train_builders:
            for _ in range(32):
                train_messages = train_builders[name]()
                eval_messages = eval_builders[name]()
                train_prompts.update(
                    message.content for message in train_messages if message.role == Role.user
                )
                eval_prompts.update(
                    message.content for message in eval_messages if message.role == Role.user
                )

    assert train_prompts
    assert eval_prompts
    assert train_prompts.isdisjoint(eval_prompts)


def test_gold_span_locates_subsequence():
    framed = [1, 2, 3, 4, 5]
    assert gold_span(framed, [3, 4]) == (2, 3)
    assert gold_span(framed, [9]) is None


def test_pointer_head_predict_span_in_range():
    import torch

    ph = PointerHead(d_model=16)
    s, e = ph.predict_span(torch.randn(7, 16), "path")
    assert 0 <= s <= e < 7
