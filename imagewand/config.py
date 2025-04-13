import configparser
import os
from pathlib import Path

# Define imagewand config directory
IMAGEWAND_CONFIG_DIR = os.path.expanduser("~/.imagewand")
# Ensure the directory exists
os.makedirs(IMAGEWAND_CONFIG_DIR, exist_ok=True)
DEFAULT_CONFIG_PATH = os.path.join(IMAGEWAND_CONFIG_DIR, "presets")


def load_presets(section="presets"):
    """
    Load presets from config file

    Args:
        section: Section name in config file ('presets' for filters,
                'rmbg_presets' for background removal)
    """
    config = configparser.ConfigParser()

    if os.path.exists(DEFAULT_CONFIG_PATH):
        config.read(DEFAULT_CONFIG_PATH)

    if section not in config:
        config[section] = {}

    return config[section]


def save_preset(name: str, preset_string: str, section="presets"):
    """
    Save a new preset to config file

    Args:
        name: Name of the preset
        preset_string: Preset configuration string
        section: Section name in config file ('presets' for filters,
                'rmbg_presets' for background removal)
    """
    config = configparser.ConfigParser()

    if os.path.exists(DEFAULT_CONFIG_PATH):
        config.read(DEFAULT_CONFIG_PATH)

    if section not in config:
        config[section] = {}

    config[section][name] = preset_string

    with open(DEFAULT_CONFIG_PATH, "w") as f:
        config.write(f)


def list_presets(section="presets"):
    """
    Return list of available presets

    Args:
        section: Section name in config file ('presets' for filters,
                'rmbg_presets' for background removal)
    """
    presets = load_presets(section)
    return {name: value for name, value in presets.items()}


def save_rmbg_preset(
    name: str,
    model: str,
    alpha_matting: bool,
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    erode_size: int = 10,
):
    """
    Save a background removal preset with the given parameters

    Args:
        name: Name of the preset
        model: Model name
        alpha_matting: Whether to use alpha matting
        foreground_threshold: Alpha matting foreground threshold
        background_threshold: Alpha matting background threshold
        erode_size: Alpha matting erode size
    """
    preset_parts = [f"model={model}"]

    if alpha_matting:
        preset_parts.append("alpha_matting=true")
        preset_parts.append(f"foreground_threshold={foreground_threshold}")
        preset_parts.append(f"background_threshold={background_threshold}")
        preset_parts.append(f"erode_size={erode_size}")

    preset_string = ",".join(preset_parts)
    save_preset(name, preset_string, "rmbg_presets")


def load_rmbg_preset(name: str):
    """
    Load a background removal preset

    Args:
        name: Name of the preset

    Returns:
        Dictionary with preset parameters or None if preset doesn't exist
    """
    presets = load_presets("rmbg_presets")
    if name not in presets:
        return None

    preset_string = presets[name]
    preset_dict = {}

    for part in preset_string.split(","):
        key, value = part.split("=")
        # Convert to appropriate types
        if key == "alpha_matting":
            preset_dict[key] = value.lower() == "true"
        elif key in ["foreground_threshold", "background_threshold", "erode_size"]:
            preset_dict[key] = int(value)
        else:
            preset_dict[key] = value

    return preset_dict
