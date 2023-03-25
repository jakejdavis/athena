import logging


def get_config_val(config, key, default=None, type=str):
    key_parts = key.split(".")
    val = config
    for key_part in key_parts:
        if key_part in val:
            val = val[key_part]
        else:
            if default is None:
                logging.error("Key %s not found in config" % key)
                exit(1)
            logging.debug("Key %s not found in config, using default value" % key)
            val = default
            break

    return type(val) if type is not None else val
