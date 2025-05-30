import configparser
import os
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from imagewand.config import (
    DEFAULT_CONFIG_PATH,
    list_presets,
    load_presets,
    save_preset,
)


@pytest.fixture
def mock_config_file():
    """Create a mock config file content with some presets."""
    config_content = """[presets]
grayscale_boost = grayscale,contrast:factor=1.8
document_scan = grayscale,sharpen:factor=2.0,contrast
photo_enhance = saturation:factor=1.2,sharpen
"""
    return config_content


def test_load_presets_no_file():
    """Test loading presets when config file doesn't exist."""
    with patch("os.path.exists", return_value=False):
        presets = load_presets()
        assert isinstance(presets, configparser.SectionProxy)
        assert len(presets) == 0


def test_load_presets_with_file(mock_config_file):
    """Test loading presets from an existing config file."""
    config = configparser.ConfigParser()
    config.read_string(mock_config_file)

    with patch("os.path.exists", return_value=True), patch(
        "configparser.ConfigParser", return_value=config
    ):
        presets = load_presets()

        assert isinstance(presets, configparser.SectionProxy)
        assert len(presets) == 3
        assert "grayscale_boost" in presets
        assert presets["grayscale_boost"] == "grayscale,contrast:factor=1.8"
        assert "document_scan" in presets
        assert "photo_enhance" in presets


def test_save_preset_new_file():
    """Test saving a preset when config file doesn't exist."""
    with patch("os.path.exists", return_value=False), patch(
        "builtins.open", mock_open()
    ) as mock_file:
        save_preset("test_preset", "grayscale,sharpen")

        mock_file.assert_called_with(DEFAULT_CONFIG_PATH, "w")
        mock_file().write.assert_called()


def test_save_preset_existing_file(mock_config_file):
    """Test saving a preset to an existing config file."""
    config = configparser.ConfigParser()
    config.read_string(mock_config_file)

    mock_file = mock_open(read_data=mock_config_file)

    with patch("os.path.exists", return_value=True), patch(
        "configparser.ConfigParser", return_value=config
    ), patch("builtins.open", mock_file):
        save_preset("new_preset", "blur,contrast")

        mock_file.assert_called_with(DEFAULT_CONFIG_PATH, "w")

        mock_file().write.assert_called()

        assert "new_preset" in config["presets"]
        assert config["presets"]["new_preset"] == "blur,contrast"


def test_list_presets(mock_config_file):
    """Test listing all available presets."""
    config = configparser.ConfigParser()
    config.read_string(mock_config_file)

    with patch("imagewand.config.load_presets", return_value=config["presets"]):
        presets = list_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 3
        assert "grayscale_boost" in presets
        assert presets["grayscale_boost"] == "grayscale,contrast:factor=1.8"
        assert "document_scan" in presets
        assert "photo_enhance" in presets


def test_integration_save_and_load():
    """Test saving and then loading a preset (integration test)."""
    temp_config_path = "/tmp/test_imagewand_config"

    with patch("imagewand.config.DEFAULT_CONFIG_PATH", temp_config_path):
        try:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            save_preset("test_preset", "grayscale,sharpen")

            presets = load_presets()

            assert "test_preset" in presets
            assert presets["test_preset"] == "grayscale,sharpen"

        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
