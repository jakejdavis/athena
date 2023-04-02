import json
import logging
import os


def load_config(config):
    """
    Load a config file from a path or a json string.

    :param config: The path to the config file or a json string
    :return: The config dictionary
    """
    if config is not None:
        if os.path.exists(config):
            with open(config, "r") as f:
                config = json.load(f)
        else:
            logging.debug(
                f"Additional config file {config} not found, trying to load as json"
            )
            config = json.loads(config)
    else:
        config = {}
    logging.info(f"Additional config: {config}")

    return config


def get_config_val(config, key, default=None, type=str):
    """
    Get a value from the config dictionary, using a key with dot notation.
    If the key is not found, return the default value.

    :param config: The config dictionary
    :param key: The key to look for, using dot notation
    :param default: The default value to return if the key is not found
    :param type: The type to cast the value to
    :return: The correctly typed value, or the default value if the key is not found
    """
    key_parts = key.split(".")
    val = config
    for key_part in key_parts:
        if key_part in val:
            val = val[key_part]
        else:
            if default is None:
                logging.debug(f"Key {key} not found in config, returning None")
                return None

            logging.debug(
                f"Key {key} not found in config, using default value {str(default)}"
            )
            val = default
            break

    return type(val) if type is not None else val
