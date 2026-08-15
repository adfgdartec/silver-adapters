"""Model adapter contracts for PyTorch, Hugging Face, and custom runtimes."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    task: str
    config: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "architecture": self.architecture, "task": self.task,
                "config": dict(self.config), "checkpoint": self.checkpoint}


class ModelAdapter(Protocol):
    def load(self, spec: ModelSpec, device: str = "cpu") -> Any: ...
    def predict(self, model: Any, batch: Any) -> Any: ...


class PyTorchModelAdapter:
    """Adapter for any user-supplied PyTorch module or Silver builder."""

    def load(self, spec: ModelSpec, device: str = "cpu") -> Any:
        if spec.architecture == "transformer":
            from silver_torch import ModelBlueprint, build_model
            model = build_model(ModelBlueprint(spec.name, spec.architecture, dict(spec.config)))
        elif spec.checkpoint:
            try:
                import torch
            except ImportError as error:
                raise ImportError("install PyTorch to load checkpoints") from error
            model = torch.jit.load(spec.checkpoint, map_location=device)
        else:
            raise ValueError("custom architectures require a checkpoint or registered adapter")
        return model.to(device) if hasattr(model, "to") else model

    def predict(self, model: Any, batch: Any) -> Any:
        return model(batch)


class HuggingFaceModelAdapter:
    """Load any AutoModel task supported by ``transformers`` without hard dependency."""

    def load(self, spec: ModelSpec, device: str = "cpu") -> Any:
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
        except ImportError as error:
            raise ImportError("install silver-adapters[huggingface] to load Hugging Face models") from error
        if not spec.checkpoint:
            raise ValueError("Hugging Face ModelSpec requires checkpoint to be a model id or path")
        loader = AutoModelForCausalLM if spec.task in ("language_modeling", "causal_lm", "gpt") else AutoModelForSequenceClassification
        return loader.from_pretrained(spec.checkpoint).to(device)

    def predict(self, model: Any, batch: Any) -> Any:
        return model(**batch) if isinstance(batch, dict) else model(batch)
