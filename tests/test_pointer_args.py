import torch


def test_browser_pointer_vocabulary_is_backward_compatible():
    from openlocalagent.agent.pointer_head import (
        BROWSER_PTR_ARGS,
        PTR_ARGS,
        PointerHead,
    )

    legacy = PointerHead(16)
    browser = PointerHead(16, args=BROWSER_PTR_ARGS)
    assert tuple(legacy.args) == tuple(PTR_ARGS)
    assert browser.arg_idx["target_id"] == len(PTR_ARGS)
    assert browser.arg_idx["value"] == len(PTR_ARGS) + 1
    assert browser.arg_emb.weight.shape == (len(BROWSER_PTR_ARGS), 16)

    features = torch.randn(7, 16)
    start, end = browser.logits(features.unsqueeze(0), torch.tensor([browser.arg_idx["target_id"]]))
    assert start.shape == end.shape == (1, 7)


def test_exported_pointer_heads_use_checkpoint_argument_metadata():
    from openlocalagent.agent.pointer_head import BROWSER_PTR_ARGS, PointerHead
    from openlocalagent.agent.tool_head import ToolHead
    from openlocalagent.inference.export.to_onnx import _heads_json

    d_model = 16
    ptr = PointerHead(d_model, args=BROWSER_PTR_ARGS)
    checkpoint = {
        "tool_head": ToolHead(d_model).state_dict(),
        "ptr_head": ptr.state_dict(),
        "ptr_args": list(BROWSER_PTR_ARGS),
    }
    heads = _heads_json(checkpoint)
    assert heads["pointer_head"]["args"] == list(BROWSER_PTR_ARGS)
    assert heads["pointer_head"]["arg_idx"]["target_id"] == len(BROWSER_PTR_ARGS) - 2
    assert len(heads["pointer_head"]["arg_emb"]) == len(BROWSER_PTR_ARGS)


def test_public_browser_names_map_only_for_legacy_auxiliary_head():
    from openlocalagent.agent.tool_head import canonical_tool_name

    assert canonical_tool_name("web_click") == "click"
    assert canonical_tool_name("web_type") == "type_text"
    assert canonical_tool_name("web_select") == "click"
    assert canonical_tool_name("send_email") == "send_email"
