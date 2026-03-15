__all__ = ["load_config"]

import json
import pathlib
import types
import typing

import jsonschema

CONFIG_PATH = pathlib.Path.home() / ".config" / "pypushgotomount"
CONFIG_FILE = CONFIG_PATH / "config.json"
CAMERA_OFFSETS_FILE = CONFIG_PATH / "camera_offsets.ini"
JSON_SCHEMA_FILE = pathlib.Path(__file__).parents[0] / "configuration_schema.json"


def load_config() -> types.SimpleNamespace:
    """Load the configuration file.

    The configuration file is expected to be at ${HOME}/.config/pypushgotomount/config.json
    If it doesn't exist then a default configuration with emulators only is loaded.

    Returns
    -------
    types.SimpleNamespace
        A SimpleNamespace containing the configuration.
    """
    config: dict[str, typing.Any] = {}

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

    with open(JSON_SCHEMA_FILE, "r") as f:
        json_schema = json.load(f)
    validator = jsonschema.Draft7Validator(schema=json_schema)
    validator.validate(config)

    configuration = types.SimpleNamespace()
    if "alt" in config:
        configuration.alt_module_name = config["alt"]["module"]
        configuration.alt_class_name = config["alt"]["class_name"]
        configuration.alt_hub_port = config["alt"]["hub_port"]
        configuration.alt_gear_reduction = config["alt"]["gear_reduction"]
    if "az" in config:
        configuration.az_module_name = config["az"]["module"]
        configuration.az_class_name = config["az"]["class_name"]
        configuration.az_hub_port = config["az"]["hub_port"]
        configuration.az_gear_reduction = config["az"]["gear_reduction"]
    if "camera" in config:
        configuration.camera_module_name = config["camera"]["module"]
        configuration.camera_class_name = config["camera"]["class_name"]
        configuration.camera_focal_length = config["camera"]["focal_length"]

    return configuration
