from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Persist model metadata for versioned model retrieval."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or "results/models")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def register(self, model_id: str, payload: dict[str, Any]) -> Path:
        """Store model metadata as a JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata_path = self.root_dir / f"{model_id}_{timestamp}.json"
        payload = {"model_id": model_id, "created_at": timestamp, **payload}
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metadata_path
