from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentManager:
    """Persist experiment metadata in a simple JSON registry."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or "results/experiments")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def register(self, experiment_name: str, metadata: dict[str, Any]) -> Path:
        """Write experiment metadata for later inspection."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = self.root_dir / f"{experiment_name}_{timestamp}.json"
        payload = {"timestamp": timestamp, **metadata}
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
