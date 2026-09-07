"""Regression tests for generic filesystem path grounding."""

from openlocalagent.agent.constrained import _arg_options, _path


def test_path_prefers_explicit_absolute_workspace_over_task_identifier() -> None:
    prompt = (
        "MCPMark filesystem task file_context/file_splitting. Create the output directory. "
        "Workspace root: /private/tmp/mcpmark-fs-transfer/workspace"
    )
    assert _path(prompt) == ["/private/tmp/mcpmark-fs-transfer/workspace"]


def test_unformatted_path_schema_uses_path_extractor() -> None:
    prompt = "Create the directory at /private/tmp/mcpmark-fs-transfer/workspace/split."
    schema = {"type": "string"}
    assert _arg_options(prompt, "path", schema, True) == [
        "/private/tmp/mcpmark-fs-transfer/workspace/split"
    ]


def test_named_directory_is_joined_to_workspace_root() -> None:
    prompt = (
        "Create a new directory named `split` in the test directory. "
        "Workspace root: /private/tmp/mcpmark-fs-transfer/workspace"
    )
    assert _path(prompt) == ["/private/tmp/mcpmark-fs-transfer/workspace/split"]


def test_markdown_named_directory_is_joined_to_workspace_root() -> None:
    prompt = (
        "**Create a new directory** named `split` in the test directory. "
        "Workspace root: /private/tmp/mcpmark-fs-transfer/workspace"
    )
    assert _path(prompt) == ["/private/tmp/mcpmark-fs-transfer/workspace/split"]


def test_main_directory_label_is_not_treated_as_child_path() -> None:
    prompt = (
        "Traverse the folder structure under the main directory and inspect the folder named "
        "`complex_structure`. Main directory: /private/tmp/mcpmark-fs-transfer/workspace"
    )
    assert _path(prompt) == ["/private/tmp/mcpmark-fs-transfer/workspace/complex_structure"]


def test_nested_quoted_parent_directory_is_preserved() -> None:
    prompt = (
        'Create a folder named `final_version` inside the folder "legal_files/" directory. '
        "Workspace root: /private/tmp/mcpmark-fs-transfer/legal_document"
    )
    assert _path(prompt) == [
        "/private/tmp/mcpmark-fs-transfer/legal_document/legal_files/final_version"
    ]
