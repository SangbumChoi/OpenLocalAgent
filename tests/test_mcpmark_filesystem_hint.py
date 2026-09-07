"""Regression tests for generic MCP filesystem operation hints."""

from openlocalagent.agent.constrained import (
    _filesystem_lexical_tool,
    _next_output_path,
    _output_filenames,
    _text_arg,
    _workspace_output_path,
)
from openlocalagent.data.schema import ToolSpec


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(name="directory_tree", description="", parameters={}),
        ToolSpec(name="write_file", description="", parameters={}),
        ToolSpec(name="create_directory", description="", parameters={}),
    ]


def test_filesystem_hint_prefers_tree_for_recursive_count() -> None:
    assert _filesystem_lexical_tool("Recursively count all .py files.", _tools()) == "directory_tree"


def test_filesystem_hint_prefers_write_for_save_instruction() -> None:
    assert _filesystem_lexical_tool("Write the answer to the output file.", _tools()) == "write_file"


def test_filesystem_hint_switches_to_write_after_tree_result() -> None:
    prompt = (
        "Recursively count the total number of .py files. Write the answer to structure_analysis.txt.\n"
        "TOOL_RESULT: [{\"name\":\"m.py\",\"type\":\"file\"}]"
    )
    assert _filesystem_lexical_tool(prompt, _tools()) == "write_file"
    assert _text_arg(prompt, "content") == ["1"]


def test_filesystem_output_path_ignores_filenames_in_tool_result() -> None:
    prompt = (
        "Write the answer to structure_analysis.txt in the main directory. Main directory: "
        "/private/tmp/mcpmark-fs-state/extracted/folder_structure\n"
        "ASSISTANT: <tool_call>{\"arguments\":{\"path\":\"/tool_call\"}}</tool_call>\n"
        "TOOL_RESULT: {\"text\": \"[{\\\"name\\\":\\\"report_2.txt\\\"}]\"}"
    )
    assert _workspace_output_path(prompt) == [
        "/private/tmp/mcpmark-fs-state/extracted/folder_structure/structure_analysis.txt"
    ]


def test_count_parser_accepts_backtick_extension_and_structured_tree() -> None:
    prompt = (
        "Count the total number of `.py` files in all subdirectories. Write the answer to "
        "structure_analysis.txt. Main directory: /tmp/workspace\n"
        "TOOL_RESULT: [{\"name\":\"a.py\",\"type\":\"file\"}, "
        "{\"name\":\"b.py\",\"type\":\"file\"}]"
    )
    assert _text_arg(prompt, "content") == ["2"]


def test_create_directory_then_empty_file_switches_to_write() -> None:
    prompt = (
        "Create a folder named `final_version`, then create an empty file named "
        "`agreement_v10.txt` inside it. Workspace root: /tmp/legal_files\n"
        "ASSISTANT: <tool_call>{\"name\":\"create_directory\",\"arguments\":{}}"
        "</tool_call>\nTOOL_RESULT: directory created"
    )
    assert _filesystem_lexical_tool(prompt, _tools()) == "write_file"
    assert _text_arg(prompt, "content") == [""]


def test_output_path_prefers_explicit_artifact_over_source_files() -> None:
    prompt = (
        "Read file_01.txt through file_20.txt and create a file named `answer.txt`. "
        "Workspace root: /tmp/file_context\nTOOL_RESULT: done"
    )
    assert _workspace_output_path(prompt) == ["/tmp/file_context/answer.txt"]


def test_output_path_preserves_parent_and_target_directories() -> None:
    prompt = (
        'Create a folder named `final_version` inside the folder "legal_files/" directory. '
        "Create an empty file with the same name as Preferred_Stock_Purchase_Agreement_v10.txt. "
        "Workspace root: /tmp/legal_document\nTOOL_RESULT: directory created"
    )
    assert _workspace_output_path(prompt) == [
        "/tmp/legal_document/legal_files/final_version/Preferred_Stock_Purchase_Agreement_v10.txt"
    ]


def test_split_read_result_is_chunked_for_next_output() -> None:
    prompt = (
        "Split large_file.txt into exactly 3 files named split_01.txt to split_03.txt. "
        "Workspace root: /tmp/file_context\n"
        "ASSISTANT: <tool_call>{\"name\":\"read_file\",\"arguments\":{}}"
        "</tool_call>\nTOOL_RESULT: "
        "{\"content\":[{\"type\":\"text\",\"text\":\"abcdefghi\"}]}"
    )
    assert _output_filenames(prompt) == ["split_01.txt", "split_02.txt", "split_03.txt"]
    assert _text_arg(prompt, "content") == ["abc"]


def test_split_output_path_starts_at_first_target_after_read() -> None:
    prompt = (
        "Split large_file.txt into exactly 3 files named split_01.txt to split_03.txt in the "
        "`split` directory. Workspace root: /tmp/file_context\n"
        "ASSISTANT: <tool_call>{\"name\":\"create_directory\",\"arguments\":{\"path\":"
        "\"/tmp/file_context/split\"}}</tool_call>\nTOOL_RESULT: ok\n"
        "ASSISTANT: <tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":"
        "\"/tmp/file_context/large_file.txt\"}}</tool_call>\nTOOL_RESULT: text"
    )
    assert _next_output_path(prompt) == "/tmp/file_context/split/split_01.txt"


def test_markdown_listed_split_targets_start_at_first_file() -> None:
    prompt = (
        "Create a new directory and split the content into exactly 3 files.\n"
        "**Name the files** as `split_01.txt`, `split_02.txt`, `split_03.txt` in the `split` directory. "
        "Workspace root: /tmp/file_context\n"
        "ASSISTANT: <tool_call>{\"name\":\"read_file\",\"arguments\":{}}"
        "</tool_call>\nTOOL_RESULT: text"
    )
    assert _next_output_path(prompt) == "/tmp/file_context/split/split_01.txt"


def test_uppercase_result_uses_matching_output_basename() -> None:
    prompt = (
        "Convert file_01.txt to uppercase and save converted files in the uppercase/ directory. "
        "Workspace root: /tmp/file_context\n"
        "ASSISTANT: <tool_call>{\"name\":\"read_multiple_files\",\"arguments\":{}}"
        "</tool_call>\nTOOL_RESULT: "
        "{\"content\":[{\"type\":\"text\",\"text\":\"/tmp/file_context/file_01.txt:\\nhello\"}]}"
    )
    assert _text_arg(prompt, "content") == ["HELLO"]


def test_duplicate_result_is_rendered_as_namesake_record() -> None:
    prompt = (
        "Find any duplicate name and generate namesake.txt. Workspace root: /tmp/student_database\n"
        "ASSISTANT: <tool_call>{\"name\":\"directory_tree\",\"arguments\":{}}"
        "</tool_call>\nTOOL_RESULT: "
        "{\"content\":[{\"type\":\"text\",\"text\":\"[DIR] 1_Ada_Lovelace\\n"
        "[DIR] 2_Ada_Lovelace\"}]}"
    )
    assert _text_arg("".join(prompt), "content") == [
        "name: Ada Lovelace\ncount: 2\nids: 1, 2"
    ]
