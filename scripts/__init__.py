"""Command-line entry points for MiniGPT experiments."""

from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    """Retain the Phase 2 report's historical Python import."""
    if name == "sample_quick":
        from scripts import phase2_samples

        return phase2_samples
    raise AttributeError(name)
