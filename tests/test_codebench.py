from openlocalagent.agent.caller import ToolCaller
from openlocalagent.eval import codebench


def test_catalog_size_and_gold():
    assert len(codebench.build_tools()) == len(codebench.CODE_TOOLS) == 24
    assert len(codebench.gold_set("eval", 7)) > 30


def test_three_arg_edit_file():
    c = ToolCaller(codebench.build_tools(), examples=codebench.examples())
    r = c.call("Replace 'foo' with 'bar' in src/app.py.")
    assert r.name == "edit_file"
    assert r.arguments == {"path": "src/app.py", "old_string": "foo", "new_string": "bar"}


def test_int_plus_quoted_arg():
    c = ToolCaller(codebench.build_tools(), examples=codebench.examples())
    r = c.call("Comment on PR 42 with 'looks good'.")
    assert r.name == "comment_on_pr"
    assert r.arguments == {"number": 42, "body": "looks good"}
    assert isinstance(r.arguments["number"], int)


def test_two_quoted_in_order():
    c = ToolCaller(codebench.build_tools(), examples=codebench.examples())
    r = c.call("Rename 'getUser' to 'fetchUser'.")
    assert r.name == "rename_symbol"
    assert r.arguments == {"old_name": "getUser", "new_name": "fetchUser"}


def test_train_eval_values_disjoint():
    for _, _, args, *_ in codebench.CODE_TOOLS:
        for _, _, tr, ev in args:
            assert not (set(tr) & set(ev))
