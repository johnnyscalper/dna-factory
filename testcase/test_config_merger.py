"""
Test cases for dna_factory/utils/config_merger.py

This module tests the configuration file merging functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.utils.config_merger import merge_config_files


class TestMergeConfigFiles:
    """Test cases for merge_config_files function"""

    def test_merge_with_only_default_config(self):
        """Test merging with only default config (no user config)"""
        # Create default config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {
                "model_name": "default_model",
                "learning_rate": 0.001,
                "batch_size": 32,
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name

        try:
            # Merge without user config
            merged_path = merge_config_files(default_path, user_config_path=None)

            # Read merged config
            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # Should be identical to default config
            assert merged == default_config

            # Cleanup
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)

    def test_merge_flat_configs(self):
        """Test merging two flat configuration files"""
        # Create default config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {
                "model_name": "default_model",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name

        # Create user config (overrides some values)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            user_config = {"model_name": "user_model", "batch_size": 64}
            yaml.safe_dump(user_config, f)
            user_path = f.name

        try:
            # Merge configs
            merged_path = merge_config_files(default_path, user_path)

            # Read merged config
            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # User values should override defaults
            assert merged["model_name"] == "user_model"
            assert merged["batch_size"] == 64
            # Default values should be preserved
            assert merged["learning_rate"] == 0.001
            assert merged["epochs"] == 10

            # Cleanup
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_nested_configs(self):
        """Test merging nested configuration structures"""
        # Create default config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {
                "model": {"name": "default_model", "size": 100, "layers": 12},
                "training": {"epochs": 10, "learning_rate": 0.001},
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name

        # Create user config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            user_config = {
                "model": {"name": "user_model", "size": 200},
                "training": {"epochs": 20},
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name

        try:
            # Merge configs
            merged_path = merge_config_files(default_path, user_path)

            # Read merged config
            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # Check nested overrides
            assert merged["model"]["name"] == "user_model"
            assert merged["model"]["size"] == 200
            # Check preserved defaults
            assert merged["model"]["layers"] == 12
            assert merged["training"]["learning_rate"] == 0.001
            # Check user overrides
            assert merged["training"]["epochs"] == 20

            # Cleanup
            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_deeply_nested_configs(self):
        """Test merging deeply nested configuration structures"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {
                "level1": {
                    "level2": {"level3": {"key1": "default1", "key2": "default2"}}
                }
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            user_config = {"level1": {"level2": {"level3": {"key1": "user1"}}}}
            yaml.safe_dump(user_config, f)
            user_path = f.name

        try:
            merged_path = merge_config_files(default_path, user_path)

            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # User override should work at deep level
            assert merged["level1"]["level2"]["level3"]["key1"] == "user1"
            # Default should be preserved
            assert merged["level1"]["level2"]["level3"]["key2"] == "default2"

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_with_list_values(self):
        """Generic keyed lists still merge; dataset overrides are tested below."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {
                "components": [
                    {"id": "ds1", "path": "/path1", "weight": 1.0},
                    {"id": "ds2", "path": "/path2", "weight": 0.5},
                ],
                "simple_list": [1, 2, 3],
            }
            yaml.safe_dump(default_config, f)
            default_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            user_config = {
                "components": [
                    {"id": "ds1", "weight": 2.0},  # Override weight for ds1
                    {"id": "ds3", "path": "/path3", "weight": 1.5},  # Add new dataset
                ],
                "simple_list": [4, 5, 6],  # Override entire list
            }
            yaml.safe_dump(user_config, f)
            user_path = f.name

        try:
            merged_path = merge_config_files(default_path, user_path)

            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # Check generic components list merge by id.
            components = merged["components"]
            ds1 = next(d for d in components if d.get("id") == "ds1")
            assert ds1["weight"] == 2.0  # Override
            assert ds1["path"] == "/path1"  # Preserved

            # Check new dataset added
            ds3 = next((d for d in components if d.get("id") == "ds3"), None)
            assert ds3 is not None
            assert ds3["weight"] == 1.5

            # Simple list should be replaced
            assert merged["simple_list"] == [4, 5, 6]

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_with_new_keys_in_user_config(self):
        """Test that new keys from user config are added"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {"existing_key": "value1"}
            yaml.safe_dump(default_config, f)
            default_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            user_config = {"new_key": "value2", "another_key": "value3"}
            yaml.safe_dump(user_config, f)
            user_path = f.name

        try:
            merged_path = merge_config_files(default_path, user_path)

            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # All keys should be present
            assert "existing_key" in merged
            assert "new_key" in merged
            assert "another_key" in merged
            assert merged["existing_key"] == "value1"
            assert merged["new_key"] == "value2"
            assert merged["another_key"] == "value3"

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_with_nonexistent_user_config(self):
        """Test that a missing explicit user config path raises FileNotFoundError"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {"key": "value"}
            yaml.safe_dump(default_config, f)
            default_path = f.name

        try:
            missing = "/nonexistent/path.yaml"
            with pytest.raises(FileNotFoundError, match="--config file not found"):
                merge_config_files(default_path, missing)
        finally:
            os.unlink(default_path)

    def test_merge_with_empty_configs(self):
        """Test merging empty configuration files"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            default_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            user_path = f.name

        try:
            merged_path = merge_config_files(default_path, user_path)

            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            # Should be empty or None
            assert merged is None or merged == {}

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_creates_valid_yaml(self):
        """Test that merged config is valid YAML"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            default_config = {"key1": "value1", "key2": {"nested": "value2"}}
            yaml.safe_dump(default_config, f)
            default_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            user_config = {"key3": "value3"}
            yaml.safe_dump(user_config, f)
            user_path = f.name

        try:
            merged_path = merge_config_files(default_path, user_path)

            # Should be able to load without error
            with open(merged_path) as f:
                merged = yaml.safe_load(f)

            assert isinstance(merged, dict)

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_preserves_unicode(self):
        """Test that Unicode characters are preserved in merge"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            default_config = {"message": "안녕하세요", "count": 42}
            yaml.safe_dump(default_config, f, allow_unicode=True)
            default_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            user_config = {"greeting": "반갑습니다", "message": "환영합니다"}
            yaml.safe_dump(user_config, f, allow_unicode=True)
            user_path = f.name

        try:
            merged_path = merge_config_files(default_path, user_path)

            with open(merged_path, encoding="utf-8") as f:
                merged = yaml.safe_load(f)

            assert merged["message"] == "환영합니다"
            assert merged["greeting"] == "반갑습니다"
            assert merged["count"] == 42

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)
            os.unlink(user_path)

    def test_merge_returns_temp_file_path(self):
        """Test that merge returns a valid temporary file path"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"key": "value"}, f)
            default_path = f.name

        try:
            merged_path = merge_config_files(default_path)

            # Should be a string
            assert isinstance(merged_path, str)
            # File should exist
            assert os.path.exists(merged_path)
            # Should be a yaml file
            assert merged_path.endswith(".yaml")
            # Should be able to read it
            with open(merged_path) as f:
                content = yaml.safe_load(f)
                assert content is not None

            os.unlink(merged_path)
        finally:
            os.unlink(default_path)


def _read_merged_config(default_path, user_config, tmp_path):
    user_path = tmp_path / "user.yaml"
    user_path.write_text(yaml.safe_dump(user_config), encoding="utf-8")
    merged_path = Path(merge_config_files(default_path, user_path))
    try:
        return yaml.safe_load(merged_path.read_text(encoding="utf-8"))
    finally:
        merged_path.unlink()


class TestDatasetListOverrides:
    """The requested dataset mixture must not inherit or collapse other data."""

    defaults = Path(__file__).resolve().parents[1] / "configs/_defaults-SDFT.yaml"

    @pytest.mark.parametrize(
        "datasets",
        [
            [
                {
                    "path": "my-org/my-prompts",
                    "columns": ["prompt"],
                    "privileged_context": "You are a concise assistant.",
                }
            ],
            [
                {
                    "path": "json",
                    "data_files": "my_rows.jsonl",
                    "split": "train",
                    "columns": ["prompt", "privileged_context"],
                }
            ],
        ],
    )
    def test_documented_custom_data_replaces_sdft_defaults(self, tmp_path, datasets):
        merged = _read_merged_config(self.defaults, {"datasets": datasets}, tmp_path)

        assert merged["datasets"] == datasets
        assert merged["teacher_model_kind"] == "base"

    def test_same_subset_name_does_not_merge_distinct_datasets(self, tmp_path):
        datasets = [
            {"path": "org/first", "name": "main", "columns": ["prompt"]},
            {"path": "org/second", "name": "main", "columns": ["messages"]},
        ]

        merged = _read_merged_config(self.defaults, {"datasets": datasets}, tmp_path)

        assert merged["datasets"] == datasets
        assert all("prompt_column" not in item for item in merged["datasets"])
        assert all(
            "privileged_context_column" not in item for item in merged["datasets"]
        )

    def test_explicit_empty_list_clears_sdft_default_mixture(self, tmp_path):
        merged = _read_merged_config(self.defaults, {"datasets": []}, tmp_path)

        assert merged["datasets"] == []

    def test_omitted_datasets_keeps_sdft_defaults(self, tmp_path):
        default_config = yaml.safe_load(self.defaults.read_text(encoding="utf-8"))

        merged = _read_merged_config(self.defaults, {"seed": 7}, tmp_path)

        assert merged["datasets"] == default_config["datasets"]
        assert merged["seed"] == 7

    def test_dataset_order_and_repeated_entries_are_preserved(self, tmp_path):
        first = {"path": "org/data", "name": "main", "split": "first"}
        second = {"path": "org/data", "name": "main", "split": "second"}
        datasets = [first, second, first]

        merged = _read_merged_config(self.defaults, {"datasets": datasets}, tmp_path)

        assert merged["datasets"] == datasets

    @pytest.mark.parametrize("key_field", ["id", "name", "key"])
    def test_unrelated_nested_keyed_lists_still_merge(self, tmp_path, key_field):
        # Only the top-level dataset mixture has replacement semantics.
        default_config = {
            "datasets": [{"path": "org/default", "name": "main"}],
            "nested": {
                "datasets": [
                    {key_field: "first", "weight": 1, "retained": True},
                    {key_field: "second", "weight": 2},
                ]
            },
        }
        default_path = tmp_path / "defaults.yaml"
        default_path.write_text(yaml.safe_dump(default_config), encoding="utf-8")
        user_config = {
            "datasets": [],
            "nested": {
                "datasets": [
                    {key_field: "first", "weight": 3},
                    {key_field: "third", "weight": 4},
                ]
            },
        }

        merged = _read_merged_config(default_path, user_config, tmp_path)

        assert merged["datasets"] == []
        assert merged["nested"]["datasets"] == [
            {key_field: "first", "weight": 3, "retained": True},
            {key_field: "second", "weight": 2},
            {key_field: "third", "weight": 4},
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
