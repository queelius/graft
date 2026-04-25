"""Config schema for the graft REST server."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Server configuration. Load from YAML via :meth:`from_yaml`."""

    llm_model: str = Field(..., description="HF model name or local path")
    infinigram_path: str = Field(..., description="Path to infinigram index directory")

    device: Optional[str] = Field(None, description="'cuda' / 'cpu' / 'mps' / None to autodetect")
    torch_dtype: Optional[str] = Field(None, description="'float16' / 'bfloat16' / 'float32' / None")

    host: str = "127.0.0.1"
    port: int = 8000

    # Default mixture parameters (overridable per-request).
    default_alpha: float = 0.3
    default_alpha_strategy: str = "constant"  # 'constant' | 'sigmoid' | 'step'

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
