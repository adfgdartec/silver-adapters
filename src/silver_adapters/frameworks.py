from typing import Any, Dict, List

from .models import CommandBridge, PythonFramework


def python_framework(
    framework: PythonFramework,
    script: str,
    args: List[str] = None,
    environment: Dict[str, str] = None,
) -> CommandBridge:
    if not script.strip():
        raise ValueError("Python bridge script is required")
    env = {"SILVER_PROTOCOL": "silver-jsonl-v1"}
    if environment:
        env.update(environment)
    return CommandBridge(
        framework=framework,
        command="python",
        args=[script] + (args or []),
        protocol="silver-jsonl-v1",
        environment=env,
    )


class PyTorchBridge:
    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        try:
            import torch

            return torch.load(checkpoint_path, map_location="cpu")
        except ImportError as error:
            raise ImportError("torch package is required for PyTorchBridge") from error

    @staticmethod
    def to_checkpoint(model: Any, optimizer: Any, epoch: int, path: str) -> None:
        try:
            import torch

            checkpoint = {"epoch": epoch}
            if hasattr(model, "state_dict"):
                checkpoint["model_state_dict"] = model.state_dict()
            if hasattr(optimizer, "state_dict"):
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(checkpoint, path)
        except ImportError as error:
            raise ImportError("torch package is required for PyTorchBridge") from error


class TensorFlowBridge:
    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> Any:
        try:
            import tensorflow as tf

            return tf.keras.models.load_model(checkpoint_path)
        except ImportError as error:
            raise ImportError(
                "tensorflow package is required for TensorFlowBridge"
            ) from error

    @staticmethod
    def to_checkpoint(model: Any, path: str) -> None:
        try:
            import tensorflow  # noqa: F401

            model.save(path)
        except ImportError as error:
            raise ImportError(
                "tensorflow package is required for TensorFlowBridge"
            ) from error
