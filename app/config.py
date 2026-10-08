"""Runtime settings. The API key comes from the environment, never from code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    jobs_db: Path
    model_dir: Path
    # One running job plus up to three waiting, as the spec requires.
    max_active_jobs: int = 4
    chunk_size: int = 250
    retention_hours: int = 6

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_key=os.environ.get("API_KEY") or None,
            jobs_db=Path(os.environ.get("JOBS_DB", ROOT / "data" / "jobs.sqlite")),
            model_dir=Path(os.environ.get("MODEL_DIR", ROOT / "model")),
        )
