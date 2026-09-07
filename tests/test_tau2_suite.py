"""tau2 is a real benchmark held outside the headline ten, on purpose."""

from openlocalagent.eval.suite import RENAME_VIEWS, SUITES, SUPPLEMENTAL_SUITES


def test_the_headline_average_is_still_ten_suites():
    """Adding an eleventh would change what every archived receipt's mean means."""
    assert len(SUITES) == 10
    assert "tau2" not in SUITES


def test_tau2_is_supplemental_and_runs_only_when_named():
    assert "tau2" in SUPPLEMENTAL_SUITES
    assert not set(SUPPLEMENTAL_SUITES) & set(SUITES)
    assert not set(SUPPLEMENTAL_SUITES) & set(RENAME_VIEWS)


def test_tau2_is_in_no_training_mixture():
    """Its score is only a transfer measurement if nothing trains on it."""
    import glob

    import yaml

    for path in glob.glob("configs/midtrain/*.yaml") + glob.glob("configs/posttrain/*.yaml"):
        text = yaml.safe_load(open(path)) or {}
        blob = str(text)
        assert "tau2" not in blob, f"{path} references tau2"
