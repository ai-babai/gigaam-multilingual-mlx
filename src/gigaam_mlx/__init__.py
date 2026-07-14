"""Native MLX inference for GigaAM multilingual CTC."""

from ._version import __version__
from .model import GigaAMCTC, load_model

__all__ = ["GigaAMCTC", "__version__", "load_model"]
