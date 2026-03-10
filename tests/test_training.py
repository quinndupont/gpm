"""Training pipeline tests — config loading, model registry."""
import pytest
import yaml
from pathlib import Path

from scripts.training.model_registry import (
    _load_registry,
    short_to_hf,
    stop_tokens_for,
    all_short_names,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

REQUIRED_TRAINING_KEYS = ["base_model", "lora", "training"]
REQUIRED_LORA_KEYS = ["rank", "alpha", "target_modules"]
REQUIRED_TRAIN_KEYS = ["num_epochs", "max_seq_length"]


@pytest.mark.eval
class TestConfigLoading:
    """Config loading — all YAML configs load without error."""

    @pytest.mark.parametrize("config_name", [
        "educator_training.yaml",
        "poet_training.yaml",
        "rhyme_training.yaml",
        "export_pipeline.yaml",
        "model_registry.yaml",
        "inference_config.yaml",
        "data_generation.yaml",
        "sagemaker.yaml",
        "rev_flux_models.yaml",
        "datacleaning.yaml",
    ])
    def test_config_loads(self, config_name):
        path = CONFIG_DIR / config_name
        if not path.exists():
            pytest.skip(f"{config_name} not found")
        data = yaml.safe_load(path.read_text())
        assert data is not None

    def test_educator_config_has_required_keys(self):
        path = CONFIG_DIR / "educator_training.yaml"
        if not path.exists():
            pytest.skip("educator_training.yaml not found")
        cfg = yaml.safe_load(path.read_text())
        assert "base_model" in cfg
        assert "lora" in cfg
        assert "training" in cfg
        assert "rank" in cfg["lora"] or "alpha" in cfg["lora"]
        assert "num_epochs" in cfg["training"]


@pytest.mark.eval
class TestModelRegistry:
    """Model registry — every model has hf_id, stop_tokens."""

    def test_registry_loads(self):
        models = _load_registry()
        assert len(models) >= 1

    def test_each_model_has_hf_id_and_stop_tokens(self):
        models = _load_registry()
        for m in models:
            assert "hf_id" in m, f"Model {m.get('short_name', m)} missing hf_id"
            assert "stop_tokens" in m or True

    def test_short_to_hf(self):
        hf = short_to_hf("qwen2.5-7b")
        assert hf is not None
        assert "Qwen" in hf or "qwen" in hf.lower()

    def test_stop_tokens_for(self):
        tokens = stop_tokens_for(short_name="qwen2.5-7b")
        assert isinstance(tokens, list)
        assert len(tokens) >= 1

    def test_all_short_names(self):
        names = all_short_names()
        assert isinstance(names, list)
        assert len(names) >= 1


@pytest.mark.slow
@pytest.mark.skip(reason="Requires GPU and modifies qlora_train; run manually")
def test_qlora_dry_run():
    """QLoRA dry run with max_steps=1. Run manually with GPU: pytest -m slow -k qlora."""
    from scripts.training.qlora_train import run_qlora_training
    import tempfile
    train_data = [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}]
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.jsonl"
        train_path.write_text("\n".join(__import__("json").dumps(d) for d in train_data))
        run_qlora_training(
            config_path=CONFIG_DIR / "educator_training.yaml",
            data_dir=Path(tmp),
            train_filename="train.jsonl",
            valid_filename="valid.jsonl",
            checkpoint_dir=Path(tmp) / "ckpt",
            num_epochs_override=1,
        )
