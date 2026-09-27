import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts/native_browsergym_eval.py"
_SPEC = importlib.util.spec_from_file_location("native_browsergym_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_model_prompt = _MODULE._model_prompt
_browser_action = _MODULE._browser_action
_dom_coordinate_candidates = _MODULE._dom_coordinate_candidates
_target_bid = _MODULE._target_bid


def test_native_browsergym_prompt_matches_train_only_adapter_contract() -> None:
    prompt = _model_prompt({"goal": "Click the button", "axtree_object": {"nodes": []}})
    assert "Browser task: Click the button" in prompt
    assert "Live accessibility elements (quoted names are valid targets):" in prompt
    assert "Choose exactly one grounded computer action or abstain." in prompt


def test_realistic_browser_tools_map_grounded_ids_to_high_level_actions() -> None:
    observation = {
        "axtree_object": {
            "nodes": [
                {"browsergym_id": "b1", "role": {"value": "button"}, "name": {"value": "Send"}},
                {"browsergym_id": "t1", "role": {"value": "textbox"}, "name": {"value": "To"}},
                {"browsergym_id": "s1", "role": {"value": "combobox"}, "name": {"value": "Label"}},
            ]
        }
    }
    assert _browser_action("web_click", {"target_id": "b1"}, observation) == ("click('b1')", True)
    assert _browser_action("web_type", {"target_id": "t1", "text": "a@b.test"}, observation) == (
        "fill('t1', 'a@b.test')",
        True,
    )
    assert _browser_action("web_select", {"target_id": "s1", "value": "Work"}, observation) == (
        "select_option('s1', 'Work')",
        True,
    )


def test_empty_accessibility_names_do_not_capture_contained_targets() -> None:
    elements = [
        {"bid": "textbox-1", "role": "textbox", "name": ""},
        {"bid": "button-1", "role": "button", "name": "Okay"},
    ]
    assert _target_bid("the Okay", elements) == "button-1"


def test_coordinate_fallback_reads_only_clickable_dom_geometry() -> None:
    observation = {
        "goal": "Click Send",
        "axtree_object": {"nodes": []},
        "dom_object": {
            "documents": [
                {
                    "strings": ["", "", "Send"],
                    "nodes": {
                        "parentIndex": [-1, 0],
                        "nodeType": [1, 3],
                        "nodeValue": [-1, 2],
                        "backendNodeId": [10, 11],
                        "attributes": [[], []],
                        "isClickable": {"index": [0]},
                    },
                    "layout": {"nodeIndex": [0], "bounds": [[20, 30, 10, 20]]},
                }
            ]
        },
    }
    candidates = _dom_coordinate_candidates(observation, device_pixel_ratio=2.0, screenshot_scale=1.5)
    assert candidates == [{"name": "Send", "x": 18.75, "y": 30.0}]
    prompt = _model_prompt(observation, candidates)
    assert '[coord-0] text: "Send"' in prompt
    assert _browser_action(
        "move_cursor",
        {"target": "Send"},
        observation,
        coordinate_fallback=True,
        device_pixel_ratio=2.0,
        screenshot_scale=1.5,
    ) == ("mouse_click(18.750, 30.000)", True)


def test_coordinate_fallback_grounds_svg_numbers_without_accessibility_roles() -> None:
    observation = {
        "goal": "Click on the numbers in ascending order.",
        "axtree_object": {"nodes": []},
        "dom_object": {
            "documents": [
                {
                    "strings": ["", "", "5", "1"],
                    "nodes": {
                        "parentIndex": [-1, 0, -1, 2],
                        "nodeType": [1, 3, 1, 3],
                        "nodeValue": [-1, 2, -1, 3],
                        "backendNodeId": [10, 11, 12, 13],
                        "attributes": [[], [], [], []],
                        "isClickable": {"index": [0, 2]},
                    },
                    "layout": {
                        "nodeIndex": [0, 2],
                        "bounds": [[0, 0, 10, 10], [10, 10, 10, 10]],
                    },
                }
            ]
        },
    }
    assert _target_bid("the numbers in ascending order", [], goal=observation["goal"]) is None
    assert _browser_action(
        "click",
        {"target": "the numbers in ascending order"},
        observation,
        coordinate_fallback=True,
    ) == ("mouse_click(15.000, 15.000)", True)
