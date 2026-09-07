from openlocalagent.track import Tracker


def test_run_and_metrics(tmp_path):
    tr = Tracker(str(tmp_path / "t"))
    rid = tr.start_run("r1", {"lr": 1e-3})
    tr.log_metric(rid, 0, "loss", 2.0)
    tr.log_metric(rid, 1, "loss", 1.5)
    tr.end_run(rid)
    assert tr.summary()["runs"] == 1


def test_content_addressed_dedup(tmp_path):
    tr = Tracker(str(tmp_path / "t"))
    a = tmp_path / "w.pt"
    a.write_bytes(b"same-weights")
    b = tmp_path / "w2.pt"
    b.write_bytes(b"same-weights")  # identical content
    c = tmp_path / "w3.pt"
    c.write_bytes(b"different")
    rid = tr.start_run("r")
    s1 = tr.log_artifact(rid, str(a), "state")
    s2 = tr.log_artifact(rid, str(b), "state")               # same bytes -> same hash, no copy
    s3 = tr.log_artifact(rid, str(c), "state")
    assert s1 == s2 != s3
    summ = tr.summary()
    assert summ["artifact_rows"] == 3 and summ["unique_blobs"] == 2
    assert summ["dedup_saved_rows"] == 1                     # one duplicate avoided


def test_latest_artifact_roundtrip(tmp_path):
    tr = Tracker(str(tmp_path / "t"))
    p = tmp_path / "d.jsonl"
    p.write_text('{"x":1}\n')
    rid = tr.start_run("r")
    tr.log_artifact(rid, str(p), "dataset")
    path = tr.latest_artifact("dataset")
    assert path and open(path).read() == '{"x":1}\n'
    assert tr.latest_artifact("nope") is None
