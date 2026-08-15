from silver_adapters import ModelSpec


def test_model_spec_is_serializable():
    spec = ModelSpec("tiny", "transformer", "classification", {"vocab_size": 10})
    assert spec.to_dict()["config"]["vocab_size"] == 10
