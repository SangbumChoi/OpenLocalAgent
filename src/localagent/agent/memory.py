"""Two-tier MemGPT/Letta-style memory (Phase 7).

  core     : small, always in-context (persona + key facts) — self-edited by the model
  archival : large, out-of-context, searchable — paged in on demand

Exposed to the model AS TOOLS (memory_append / memory_search / memory_replace) so memory
management is itself agent behavior. A consolidation policy moves stale core facts to archival.
"""

from __future__ import annotations


class Memory:
    def __init__(self, store_path: str | None = None):
        self.core: list[str] = []
        self.store_path = store_path  # archival backend (SQLite/vector) — TODO(phase-7/8)

    # --- exposed to the model as tools ---
    def memory_append(self, text: str) -> dict:
        raise NotImplementedError("TODO(phase-7): append to core, page out if over budget")

    def memory_search(self, query: str, k: int = 3) -> dict:
        raise NotImplementedError("TODO(phase-7): search archival store")

    def memory_replace(self, old: str, new: str) -> dict:
        raise NotImplementedError("TODO(phase-7): self-edit a core fact")

    # --- runtime hooks ---
    def render_core(self) -> str:
        """Block injected into the system prompt each turn."""
        return "\n".join(self.core)

    def consolidate(self) -> None:
        raise NotImplementedError("TODO(phase-7): page/consolidate core -> archival")
