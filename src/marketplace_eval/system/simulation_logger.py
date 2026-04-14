from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class StepRecord:
    t: int
    tau: int | None
    user_id: str | None
    qid: str | None  # Question ID for tracking
    user_question: str | None  # The question asked by the user
    ground_truth_answer: str | None  # Ground truth answer (if available in user data)
    generator_id: str | None
    generator_response: str | None
    score: (
        float | None
    )  # Numeric score from judge (e.g., 1.0 for correct, 0.0 otherwise)
    judge_grade: str | None  # Raw grade from judge (e.g., "A", "B", "C")
    preference_distribution: (
        str | None
    )  # JSON string of user's preference distribution after update
    router_id: str | None = None
    router_response: str | None = None
    retriever_id: str | None = None
    retrieval_result: str | None = None


class SimulationLogger:
    """Logger that stores step records and supports incremental CSV appending."""

    def __init__(self, csv_path: str | Path | None = None):
        self.step_records: List[StepRecord] = []
        self._csv_path: Path | None = Path(csv_path) if csv_path else None
        self._csv_header_written = False

    def set_csv_path(self, path: str | Path):
        """Set or update the CSV output path."""
        self._csv_path = Path(path)
        self._csv_header_written = False

    def log_step(self, step_record: StepRecord | Iterable[StepRecord]):
        """Log one or more step records and append to CSV if configured."""
        if isinstance(step_record, StepRecord):
            records = [step_record]
        else:
            records = list(step_record)

        self.step_records.extend(records)

        # Incrementally append to CSV
        if self._csv_path is not None:
            self._append_to_csv(records)

    def _append_to_csv(self, records: List[StepRecord]):
        """Append records to the CSV file, writing header on first call."""
        if not records:
            return

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [f.name for f in fields(StepRecord)]

        write_header = not self._csv_header_written and not self._csv_path.exists()
        # If file doesn't exist yet, always write header
        if not self._csv_path.exists():
            write_header = True

        mode = "a" if self._csv_header_written else "w"
        with self._csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header or not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            for record in records:
                writer.writerow(asdict(record))

    def to_db(self, table_name: str):
        raise NotImplementedError(
            "Database logging is not implemented in the prototype."
        )

    def to_df(self):
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required for DataFrame export.") from exc
        return pd.DataFrame([asdict(record) for record in self.step_records])

    def to_csv(self, path: str | Path):
        df = self.to_df()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def to_json(self, path: str | Path):
        df = self.to_df()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient="records", indent=2)
