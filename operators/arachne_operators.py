import logging

from keras.engine.sequential import Sequential
from numpy import ndarray

from utils.config import get_config_val

from .arachne import localisation, searchers
from .arachne.searchers.de_multiprocessing import apply_patch


def operator_arachne(
    model: Sequential,
    pos: tuple,
    neg: tuple,
    pos_trivial: tuple = None,
    neg_trivial: tuple = None,
    additional_config: dict = None,
) -> None:
    assert model is not None, "Model not found"
    assert len(pos[0]) > 0, "No positive examples"
    assert len(neg[0]) > 0, "No negative examples"
    assert len(neg[0]) >= len(pos[0]), "More positive examples than negative examples"

    localisation_method = get_config_val(
        additional_config, "operator.localisation", "bidirectional"
    )
    if localisation_method == "bidirectional":
        logging.info(f"Localising patch using bidirectional localisation")
        patch, _ = localisation.bidirectional_localisation(model, pos, neg)
        logging.info("Patch localised: %s" % patch)
    elif localisation_method == "random":
        logging.info(f"Localising patch using random localisation")
        patch = localisation.random_localisation(model, pos, neg)
        logging.info("Patch localised: %s" % patch)
    else:
        logging.error(f"Localisation method {localisation_method} not implemented")

    # Set searcher from config
    if get_config_val(additional_config, "operator.searcher.name", "de") == "de":
        logging.info(f"Generating patch using differential evolution")
        workers = get_config_val(additional_config, "operator.searcher.workers", 1, int)

        if workers != 1:
            logging.debug(
                f"Using multiprocessing DE with {'max' if workers == -1 else workers} workers"
            )
            patch_searcher = searchers.DE_MP(
                model,
                (pos, neg),
                (pos_trivial, neg_trivial),
                patch,
                additional_config,
                workers,
            )
        else:
            patch_searcher = searchers.DE(
                model, (pos, neg), (pos_trivial, neg_trivial), patch, additional_config
            )

    patched_weights = patch_searcher.search().x
    patched_model = apply_patch(model, patched_weights, patch)
    logging.info("Patch generated: %s" % patched_weights)

    return patched_model
