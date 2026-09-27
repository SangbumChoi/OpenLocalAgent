"""Make an exported bundle loadable by `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`.

The bundle already had HF's *file layout* - config.json, model.safetensors, tokenizer.json, a model
card. What it lacked was the two things `Auto*` actually dispatches on: an `architectures` entry and
an `auto_map` pointing at modeling code. Without them a reader has to clone this repository to load
a model we describe as fully open, which is the opposite of the claim.

The modeling code is *vendored into the bundle* rather than imported from `openlocalagent`, so the
bundle stands alone: `transformer.py` needs only `config.py` and `vision.py`, and HF's dynamic
module loader supports relative imports between shipped files. A user with `transformers` and
`torch` can load it; nothing else is required.

This is the one place `transformers` enters the project, and it enters only the *exported artifact*
- never the training path, which stays pure PyTorch. The cost is that the wrapper has to track the
architecture: `tests/test_hf_auto_bundle.py` loads a bundle through `Auto*` and asserts the logits
equal the original model's, so a drift shows up as a failing test rather than as a silently wrong
model on the Hub.
"""

from __future__ import annotations

import re
from pathlib import Path

MODEL_PACKAGE = Path(__file__).resolve().parents[2] / "model"

#: Vendored as flat modules so HF's dynamic loader can import them relatively.
VENDORED = {
    "config.py": "localagent_config.py",
    "vision.py": "localagent_vision.py",
    "transformer.py": "localagent_transformer.py",
}

CONFIGURATION = '''"""Vendored config for the exported bundle - do not edit here."""

from transformers import PretrainedConfig

from .localagent_config import ModelConfig


class LocalAgentConfig(PretrainedConfig):
    """PretrainedConfig view of this project's ModelConfig.

    Every ModelConfig field is carried verbatim so the dataclass can be rebuilt exactly; nothing
    is renamed to look more like a standard architecture, because a renamed field is a field that
    silently stops matching the checkpoint.
    """

    model_type = "openlocalagent"

    def __init__(self, **kwargs):
        known = set(ModelConfig.__dataclass_fields__)
        self.openlocalagent = {k: v for k, v in kwargs.items() if k in known}
        # transformers >= 5 validates a few attribute names of its own (`layer_types` must be one
        # of its block kinds); this project's values live in `self.openlocalagent` and are
        # rebuilt from there, so those names are not mirrored as attributes.
        for key, value in self.openlocalagent.items():
            if key in {"layer_types"}:
                continue
            setattr(self, key, value)
        # Names transformers itself reads.
        self.vocab_size = self.openlocalagent.get("vocab_size", 16384)
        self.max_position_embeddings = self.openlocalagent.get("max_seq_len", 2048)
        self.hidden_size = self.openlocalagent.get("d_model", 512)
        self.num_hidden_layers = self.openlocalagent.get("n_layers", 8)
        self.num_attention_heads = self.openlocalagent.get("n_heads", 8)
        super().__init__(**{k: v for k, v in kwargs.items() if k not in known})

    def model_config(self) -> ModelConfig:
        return ModelConfig(**self.openlocalagent)
'''

MODELING = '''"""Vendored modeling for the exported bundle - do not edit here.

Wraps this project's LocalAgentLM, which is pure PyTorch, in the minimum PreTrainedModel surface
that AutoModelForCausalLM needs. The inner module is named `model` and `base_model_prefix` matches
it, which is what lets HF load an unprefixed checkpoint into the wrapper.
"""

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_openlocalagent import LocalAgentConfig
from .localagent_transformer import LocalAgentLM


class LocalAgentForCausalLM(PreTrainedModel):
    config_class = LocalAgentConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = False
    # LocalAgentLM ties the LM head to the input embedding whenever cfg.lm_head is absent, which
    # is every tier here. transformers needs that declared or it treats the head as a missing
    # weight and tries to move it off the meta device.
    _tied_weights_keys = {"model.lm_head.weight": "model.embed.weight"}

    def __init__(self, config: LocalAgentConfig):
        super().__init__(config)
        self.model = LocalAgentLM(config.model_config())
        # Not optional in transformers 5.x: post_init builds the tied-weight bookkeeping that
        # from_pretrained consults after loading. Without it the load fails on
        # `all_tied_weights_keys` *after* every tensor has already been read.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        logits, loss = self.model(input_ids, targets=labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, **kwargs):
        """Greedy decode. The repository's KV-cache path is the fast one; this is the portable one."""
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.config.max_position_embeddings:]
            logits, _ = self.model(window)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids
'''


def vendor_modeling_code(out_path: Path) -> dict[str, str]:
    """Copy the model package into the bundle with imports rewritten, and write the HF wrappers.

    Returns the `auto_map` for config.json. Rewriting `from openlocalagent.model.x import` to
    `from .localagent_x import` is what makes the bundle standalone - with the original imports it
    would load only for someone who already has this repository installed.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in VENDORED.items():
        text = (MODEL_PACKAGE / source_name).read_text(encoding="utf-8")
        text = re.sub(r"from openlocalagent\.model\.(\w+) import", r"from .localagent_\1 import", text)
        text = re.sub(r"from openlocalagent\.model import", "from .localagent_transformer import", text)
        header = (f"# Vendored from openlocalagent/model/{source_name} by "
                  "openlocalagent.inference.export.hf_auto.\n"
                  "# Edit the source, re-export the bundle; do not patch this copy.\n")
        (out_path / target_name).write_text(header + text, encoding="utf-8")
    (out_path / "configuration_openlocalagent.py").write_text(CONFIGURATION, encoding="utf-8")
    (out_path / "modeling_openlocalagent.py").write_text(MODELING, encoding="utf-8")
    return {
        "AutoConfig": "configuration_openlocalagent.LocalAgentConfig",
        "AutoModel": "modeling_openlocalagent.LocalAgentForCausalLM",
        "AutoModelForCausalLM": "modeling_openlocalagent.LocalAgentForCausalLM",
    }
