import configparser
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.imagewand_presets")


def load_presets():
    """Load filter presets from config file"""
    config = configparser.ConfigParser()

    if os.path.exists(DEFAULT_CONFIG_PATH):
        config.read(DEFAULT_CONFIG_PATH)

    if "presets" not in config:
        config["presets"] = {}

    return config["presets"]


def save_preset(name: str, filter_string: str):
    """Save a new preset to config file"""
    config = configparser.ConfigParser()

    if os.path.exists(DEFAULT_CONFIG_PATH):
        config.read(DEFAULT_CONFIG_PATH)

    if "presets" not in config:
        config["presets"] = {}

    config["presets"][name] = filter_string

    with open(DEFAULT_CONFIG_PATH, "w") as f:
        config.write(f)


def list_presets():
    """Return list of available presets"""
    presets = load_presets()
    return {name: value for name, value in presets.items()}
