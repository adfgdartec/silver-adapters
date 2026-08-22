import pytest
import builtins
from silver_adapters import (
    PythonFramework,
    CommandBridge,
    DataImportPlan,
    SilverJsonlEvent,
    NotebookCell,
    NotebookDocument,
    python_framework,
    pandas_import,
    kaggle_import,
    encode_jsonl,
    decode_jsonl,
    notebook_document,
    silver_notebook_cells,
    RemoteTrainingClient,
    PyTorchBridge,
    TensorFlowBridge,
)


class TestPythonFramework:
    def test_framework_enum(self):
        assert PythonFramework.PYTORCH.value == "pytorch"
        assert PythonFramework.KERAS.value == "keras"
        assert PythonFramework.TENSORFLOW.value == "tensorflow"


class TestCommandBridge:
    def test_command_bridge_creation(self):
        bridge = CommandBridge(
            framework=PythonFramework.PYTORCH,
            command="python",
            args=["train.py"],
            protocol="silver-jsonl-v1",
            environment={"VAR": "value"},
        )
        assert bridge.framework == PythonFramework.PYTORCH
        assert bridge.command == "python"
        assert bridge.args == ["train.py"]
        assert bridge.protocol == "silver-jsonl-v1"
        assert bridge.environment == {"VAR": "value"}

    def test_python_framework_function(self):
        bridge = python_framework(
            framework=PythonFramework.PYTORCH,
            script="train.py",
            args=["--config", "config.yaml"],
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )
        assert bridge.framework == PythonFramework.PYTORCH
        assert bridge.command == "python"
        assert bridge.args == ["train.py", "--config", "config.yaml"]
        assert "SILVER_PROTOCOL" in bridge.environment
        assert bridge.environment["CUDA_VISIBLE_DEVICES"] == "0"

    def test_python_framework_empty_script_raises_error(self):
        with pytest.raises(ValueError, match="Python bridge script is required"):
            python_framework(PythonFramework.PYTORCH, "")


class TestDataImportPlan:
    def test_pandas_import_creation(self):
        plan = pandas_import("data/train.csv", {"sep": ","})
        assert plan.source == "pandas"
        assert plan.command == "python"
        assert "silver_pandas_bridge" in plan.args
        assert any("train.csv" in argument for argument in plan.args)
        assert plan.options == {"sep": ","}

    def test_pandas_import_empty_path_raises_error(self):
        with pytest.raises(ValueError, match="pandas source path is required"):
            pandas_import("")

    def test_kaggle_import_creation(self):
        plan = kaggle_import("competiton/dataset-name")
        assert plan.source == "kaggle"
        assert plan.command == "kaggle"
        assert "datasets" in plan.args
        assert "download" in plan.args
        assert "competiton/dataset-name" in plan.args

    def test_kaggle_import_empty_dataset_raises_error(self):
        with pytest.raises(ValueError, match="Kaggle dataset identifier is required"):
            kaggle_import("")


class TestSilverJsonlEvent:
    def test_jsonl_event_creation(self):
        event = SilverJsonlEvent(kind="epoch", data={"epoch": 1, "loss": 0.5})
        assert event.kind == "epoch"
        assert event.data == {"epoch": 1, "loss": 0.5}

    def test_jsonl_event_default_data(self):
        event = SilverJsonlEvent(kind="test")
        assert event.data == {}


class TestJsonlEncoding:
    def test_empty_event_stream_encodes_to_empty_text(self):
        assert encode_jsonl([]) == ""

    def test_event_kind_cannot_be_overridden_by_data(self):
        encoded = encode_jsonl([SilverJsonlEvent("metric", {"kind": "spoofed"})])
        assert decode_jsonl(encoded)[0].kind == "metric"

    @pytest.mark.parametrize("kind", ["", "   ", 42])
    def test_event_kind_must_be_a_non_empty_string(self, kind):
        with pytest.raises(ValueError, match="kind"):
            encode_jsonl([SilverJsonlEvent(kind, {})])

    @pytest.mark.parametrize("payload", ['{"kind": ""}', '{"kind": 42}'])
    def test_decoded_event_kind_must_be_a_non_empty_string(self, payload):
        with pytest.raises(ValueError, match="kind"):
            decode_jsonl(payload)

    def test_encode_jsonl(self):
        events = [
            SilverJsonlEvent(kind="epoch", data={"epoch": 1}),
            SilverJsonlEvent(kind="metrics", data={"accuracy": 0.9}),
        ]
        encoded = encode_jsonl(events)

        lines = encoded.strip().split("\n")
        assert len(lines) == 2
        assert '"kind": "epoch"' in lines[0]
        assert '"kind": "metrics"' in lines[1]

    def test_decode_jsonl(self):
        jsonl = '{"kind": "epoch", "epoch": 1}\n{"kind": "metrics", "accuracy": 0.9}\n'
        events = decode_jsonl(jsonl)

        assert len(events) == 2
        assert events[0].kind == "epoch"
        assert events[0].data == {"epoch": 1}
        assert events[1].kind == "metrics"
        assert events[1].data == {"accuracy": 0.9}

    def test_decode_jsonl_handles_empty_lines(self):
        jsonl = '{"kind": "test"}\n\n{"kind": "test2"}\n'
        events = decode_jsonl(jsonl)
        assert len(events) == 2

    def test_decode_jsonl_invalid_json_raises_error(self):
        jsonl = '{"kind": "test"}\ninvalid json\n'
        with pytest.raises(ValueError, match="invalid JSON"):
            decode_jsonl(jsonl)

    def test_decode_jsonl_missing_kind_raises_error(self):
        jsonl = '{"data": "test"}\n'
        with pytest.raises(ValueError, match="must be an object with 'kind' field"):
            decode_jsonl(jsonl)

    def test_decode_jsonl_non_object_raises_error(self):
        jsonl = "[1, 2, 3]\n"
        with pytest.raises(ValueError, match="must be an object with 'kind' field"):
            decode_jsonl(jsonl)

    def test_encode_decode_roundtrip(self):
        original_events = [
            SilverJsonlEvent(kind="epoch", data={"epoch": 1, "loss": 0.5}),
            SilverJsonlEvent(kind="metrics", data={"accuracy": 0.9}),
        ]
        encoded = encode_jsonl(original_events)
        decoded = decode_jsonl(encoded)

        assert len(decoded) == len(original_events)
        for orig, dec in zip(original_events, decoded):
            assert orig.kind == dec.kind
            assert orig.data == dec.data


class TestNotebookCell:
    def test_notebook_cell_creation(self):
        cell = NotebookCell(
            cell_type="code",
            metadata={"language": "python"},
            source=["print('hello')"],
            outputs=[],
            execution_count=1,
        )
        assert cell.cell_type == "code"
        assert cell.metadata == {"language": "python"}
        assert cell.source == ["print('hello')"]
        assert cell.outputs == []
        assert cell.execution_count == 1


class TestNotebookDocument:
    def test_notebook_document_creation(self):
        cells = [NotebookCell(cell_type="markdown", metadata={}, source=["# Title"])]
        doc = notebook_document(cells)

        assert doc.nbformat == 4
        assert doc.nbformat_minor == 5
        assert len(doc.cells) == 1
        assert "kernelspec" in doc.metadata

    def test_silver_notebook_cells(self):
        cells = silver_notebook_cells()

        assert len(cells) == 3
        assert cells[0].cell_type == "markdown"
        assert cells[1].cell_type == "code"
        assert cells[2].cell_type == "code"

        # Check that silver training bridge is mentioned
        assert any("Silver training bridge" in " ".join(cell.source) for cell in cells)


class TestRemoteTrainingClient:
    def test_client_timeout_is_configurable(self):
        client = RemoteTrainingClient("https://api.example.com", timeout=4.5)
        assert client.timeout == 4.5
        with pytest.raises(ValueError, match="timeout must be positive"):
            RemoteTrainingClient("https://api.example.com", timeout=0)

    def test_client_creation(self):
        try:
            import requests

            client = RemoteTrainingClient("https://api.example.com")
            assert client.base_url == "https://api.example.com"
            assert client.headers == {}
        except ImportError:
            pytest.skip("requests not installed")

    def test_client_trailing_url_removed(self):
        try:
            import requests

            client = RemoteTrainingClient("https://api.example.com/")
            assert client.base_url == "https://api.example.com"
        except ImportError:
            pytest.skip("requests not installed")

    def test_client_custom_headers(self):
        try:
            import requests

            client = RemoteTrainingClient(
                "https://api.example.com", headers={"Authorization": "Bearer token"}
            )
            assert client.headers["Authorization"] == "Bearer token"
        except ImportError:
            pytest.skip("requests not installed")

    def test_client_empty_url_raises_error(self):
        with pytest.raises(ValueError, match="Remote training base URL is required"):
            RemoteTrainingClient("")

    def test_client_without_requests_raises_error(self):
        # Mock the import error
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("requests not available")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="requests package is required"):
                RemoteTrainingClient("https://api.example.com")
        finally:
            builtins.__import__ = original_import


class TestPyTorchBridge:
    def test_pytorch_bridge_without_torch_raises_error(self):
        # Mock the import error
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not available")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="torch package is required"):
                PyTorchBridge.from_checkpoint("model.pt")
        finally:
            builtins.__import__ = original_import

    def test_pytorch_bridge_to_checkpoint_without_torch_raises_error(self):
        # Mock the import error
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not available")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="torch package is required"):
                PyTorchBridge.to_checkpoint(None, None, 1, "model.pt")
        finally:
            builtins.__import__ = original_import


class TestTensorFlowBridge:
    def test_tensorflow_bridge_without_tensorflow_raises_error(self):
        # Mock the import error
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise ImportError("tensorflow not available")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="tensorflow package is required"):
                TensorFlowBridge.from_checkpoint("model.h5")
        finally:
            builtins.__import__ = original_import

    def test_tensorflow_bridge_to_checkpoint_without_tensorflow_raises_error(self):
        # Mock the import error
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise ImportError("tensorflow not available")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="tensorflow package is required"):
                TensorFlowBridge.to_checkpoint(None, "model.h5")
        finally:
            builtins.__import__ = original_import


class TestEnvironmentVariableHandling:
    def test_python_framework_default_environment(self):
        bridge = python_framework(PythonFramework.PYTORCH, "train.py")
        assert "SILVER_PROTOCOL" in bridge.environment
        assert bridge.environment["SILVER_PROTOCOL"] == "silver-jsonl-v1"

    def test_python_framework_custom_environment(self):
        bridge = python_framework(
            PythonFramework.PYTORCH,
            "train.py",
            environment={"CUSTOM_VAR": "custom_value"},
        )
        assert "SILVER_PROTOCOL" in bridge.environment
        assert bridge.environment["CUSTOM_VAR"] == "custom_value"


class TestDataImportOptions:
    def test_pandas_import_with_options(self):
        plan = pandas_import("data.csv", {"sep": ",", "encoding": "utf-8"})
        assert plan.options == {"sep": ",", "encoding": "utf-8"}

    def test_pandas_import_default_options(self):
        plan = pandas_import("data.csv")
        assert plan.options == {}

    def test_kaggle_import_with_options(self):
        plan = kaggle_import("dataset/name", {"unzip": True})
        assert plan.options == {"unzip": True}
