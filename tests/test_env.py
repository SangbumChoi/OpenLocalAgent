import os

from openlocalagent.env import load_env_file


def test_load_env_file_fills_missing_keys_without_overriding_process_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# ignored comment\nLOCALAGENT_TEST_TOKEN='secret-looking-value'\n"
        "LOCALAGENT_TEST_EXISTING=file-value\nINVALID-KEY=nope\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCALAGENT_TEST_EXISTING", "process-value")

    load_env_file(env_file)

    assert os.environ["LOCALAGENT_TEST_TOKEN"] == "secret-looking-value"
    assert os.environ["LOCALAGENT_TEST_EXISTING"] == "process-value"
    assert "INVALID-KEY" not in os.environ
