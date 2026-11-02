"""事件流：JSONL，逐回合可重放（不变量 I6）。"""

from __future__ import annotations

import json
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def emit(self, turn: int, event: str, **payload) -> dict:
        record = {"turn": turn, "event": event, **payload}
        self.events.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def close(self) -> None:
        return None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
