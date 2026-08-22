import json
from typing import Dict, List

from .models import DataImportPlan, SilverJsonlEvent


def pandas_import(path: str, options: Dict[str, str] = None) -> DataImportPlan:
    if not path.strip():
        raise ValueError("pandas source path is required")
    return DataImportPlan(
        source="pandas",
        command="python",
        args=["-m", "silver_pandas_bridge", path],
        output="jsonl",
        options=options or {},
    )


def kaggle_import(dataset: str, options: Dict[str, str] = None) -> DataImportPlan:
    if not dataset.strip():
        raise ValueError("Kaggle dataset identifier is required")
    return DataImportPlan(
        source="kaggle",
        command="kaggle",
        args=["datasets", "download", "-d", dataset, "--unzip"],
        output="jsonl",
        options=options or {},
    )


def encode_jsonl(events: List[SilverJsonlEvent]) -> str:
    if not events:
        return ""
    encoded = []
    for event in events:
        if not isinstance(event.kind, str) or not event.kind.strip():
            raise ValueError("JSONL event kind must be a non-empty string")
        encoded.append(json.dumps({**event.data, "kind": event.kind}))
    return "\n".join(encoded) + "\n"


def decode_jsonl(text: str) -> List[SilverJsonlEvent]:
    events = []
    for line_number, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or "kind" not in data:
                raise ValueError(
                    f"Line {line_number}: must be an object with 'kind' field"
                )
            kind = data.pop("kind")
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError(
                    f"Line {line_number}: 'kind' must be a non-empty string"
                )
            events.append(SilverJsonlEvent(kind=kind, data=data))
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {line_number}: invalid JSON - {error}") from error
    return events
