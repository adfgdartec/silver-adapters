from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class PythonFramework(Enum):
    PYTORCH = "pytorch"
    KERAS = "keras"
    TENSORFLOW = "tensorflow"


@dataclass(frozen=True)
class CommandBridge:
    framework: PythonFramework
    command: str
    args: List[str]
    protocol: str = "silver-jsonl-v1"
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataImportPlan:
    source: Literal["pandas", "kaggle"]
    command: str
    args: List[str]
    output: str = "jsonl"
    options: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SilverJsonlEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotebookCell:
    cell_type: Literal["markdown", "code"]
    metadata: Dict[str, Any]
    source: List[str]
    outputs: Optional[List[Any]] = None
    execution_count: Optional[int] = None


@dataclass(frozen=True)
class NotebookDocument:
    cells: List[NotebookCell]
    metadata: Dict[str, Any]
    nbformat: int = 4
    nbformat_minor: int = 5
