"""The route taxonomy must stay in lock-step with the concrete tool pool: every tool the model can
emit maps to exactly one of the 5 stable routes, so the route head never silently mis-buckets a new
tool. This is the guard that keeps the small closed set honest as the tool pool grows."""

from openlocalagent.agent.routes import ROUTES, route_of, route_of_sample
from openlocalagent.agent.tool_head import CLASSES


def test_five_stable_routes():
    assert ROUTES == ["web_search", "computer_use", "code", "app_action", "text"]


def test_every_tool_maps_to_a_route():
    # full coverage: no concrete tool falls through to the default
    for name in CLASSES:
        assert route_of(name) in ROUTES, name
    # "text" tool name routes to the text modality
    assert route_of("text") == "text"


def test_unknown_tool_falls_back_to_text():
    # out-of-pool tool name must not crash — degrades to a direct answer
    assert route_of("some_unseen_tool_xyz") == "text"


def test_route_of_sample_uses_kind_and_ref_name():
    class S:
        def __init__(self, kind, ref_name=""):
            self.kind = kind
            self.ref_name = ref_name

    assert route_of_sample(S("text")) == "text"
    assert route_of_sample(S("tool", "click")) == "computer_use"
    assert route_of_sample(S("tool", "web_search")) == "web_search"
    assert route_of_sample(S("tool", "send_email")) == "app_action"
    assert route_of_sample(S("tool", "run_python")) == "code"


def test_routes_are_a_strict_coarsening():
    # many tools, few routes — the whole point. Each route is non-empty.
    covered = {route_of(n) for n in CLASSES}
    assert covered == set(ROUTES)
    assert len(ROUTES) < len(CLASSES)
