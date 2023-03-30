import logging


def get_config_val(config, key, default=None, type=str):
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
