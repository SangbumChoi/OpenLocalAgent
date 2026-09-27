"""An exported bundle must load through Auto* and produce the model it claims to be.

The bundle vendors a copy of the model package, so the wrapper can drift from the architecture it
wraps - and a drifted wrapper does not crash, it returns a different model. That is the failure
worth a test: not "does it load" but "are the logits identical".

transformers appears only here and in the exported artifact. The training path stays pure PyTorch.
"""

from pathlib import Path

import pytest
import torch

from openlocalagent.inference.export.to_hf import export_hf
from openlocalagent.model import LocalAgentLM, ModelConfig

pytest.importorskip("transformers")
pytest.importorskip("safetensors")


def tiny_config() -> ModelConfig:
    return ModelConfig(name="tiny", vocab_size=256, d_model=32, n_layers=2,
                       n_heads=4, n_kv_heads=2, ffn_hidden=48, max_seq_len=64)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> tuple[Path, LocalAgentLM]:
    torch.manual_seed(0)
    cfg = tiny_config()
    model = LocalAgentLM(cfg).eval()
    root = tmp_path_factory.mktemp("hf")
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "step": 0},
               root / "latest.pt")
    export_hf(str(root / "latest.pt"), str(root / "bundle"))
    return root / "bundle", model


def test_bundle_ships_what_auto_dispatches_on(bundle):
    import json

    path, _ = bundle
    config = json.loads((path / "config.json").read_text())
    assert config["architectures"] == ["LocalAgentForCausalLM"]
    assert config["auto_map"]["AutoModelForCausalLM"].endswith("LocalAgentForCausalLM")
    assert config["model_type"] == "openlocalagent"


def test_bundle_is_standalone(bundle):
    """No file may import `openlocalagent`, or the bundle loads only for someone who has this repo."""
    path, _ = bundle
    import ast

    for source in path.glob("*.py"):
        # Parse rather than grep: the vendoring header names the file it copied from, so a
        # substring check flags its own provenance comment as an import.
        for node in ast.walk(ast.parse(source.read_text())):
            # Only ABSOLUTE imports of the package matter. `from .localagent_transformer import`
            # is a relative import of a vendored sibling - the very thing that makes the bundle
            # standalone - and it starts with the same letters.
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                assert module != "openlocalagent" and not module.startswith("openlocalagent."), (
                    f"{source.name} imports {module} from the repository package")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "openlocalagent" \
                        and not alias.name.startswith("openlocalagent."), (
                        f"{source.name} imports {alias.name} from the repository package")
    for required in ("modeling_openlocalagent.py", "configuration_openlocalagent.py",
                     "localagent_transformer.py", "localagent_config.py",
                     "model.safetensors", "generation_config.json"):
        assert (path / required).is_file(), f"bundle is missing {required}"


def test_automodel_reproduces_the_original_logits(bundle):
    """The assertion that matters: a drifted wrapper returns a different model, not an error."""
    from transformers import AutoModelForCausalLM

    path, original = bundle
    loaded = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True).eval()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        reference, _ = original(ids)
        produced = loaded(input_ids=ids).logits
    assert produced.shape == reference.shape
    assert torch.equal(produced, reference), "Auto* model diverged from the exported checkpoint"


def test_labels_produce_a_loss(bundle):
    from transformers import AutoModelForCausalLM

    path, _ = bundle
    loaded = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True).eval()
    ids = torch.randint(0, 256, (1, 16))
    out = loaded(input_ids=ids, labels=ids)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_generate_appends_tokens(bundle):
    from transformers import AutoModelForCausalLM

    path, _ = bundle
    loaded = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True).eval()
    ids = torch.randint(0, 256, (1, 4))
    assert loaded.generate(ids, max_new_tokens=3).shape == (1, 7)


def test_autoconfig_round_trips_the_model_config(bundle):
    from transformers import AutoConfig

    path, original = bundle
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    round_tripped = config.model_config()
    # The bundle's ModelConfig is a *vendored copy*, so it is a different class and dataclass
    # __eq__ returns NotImplemented however identical the fields are. Compare structurally, which
    # is also the only comparison that means anything here.
    differing = {
        name: (getattr(original.cfg, name), getattr(round_tripped, name))
        for name in original.cfg.__dataclass_fields__
        if getattr(original.cfg, name) != getattr(round_tripped, name)
    }
    assert not differing, f"ModelConfig changed across the round trip: {differing}"
