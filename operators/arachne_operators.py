from .arachne import localisation
from .arachne import searchers
from keras.engine.sequential import Sequential
from numpy import ndarray
import logging


def operator_arachne(model: Sequential, pos: tuple, neg: tuple) -> None:
    assert model is not None, "Model not found"
    assert len(pos[0]) > 0, "No positive examples"
    assert len(neg[0]) > 0, "No negative examples"
    assert len(neg[0]) >= len(pos[0]), "More positive examples than negative examples"

    pareto, _ = localisation.bidirectional_localisation(model, pos, neg)
    patch_searcher = searchers.DE(model, (pos, neg), pareto)
    patch = patch_searcher.search().x
    patched_model = patch_searcher.apply_patch(patch)
    logging.info("Patch found: %s" % patch)

    return patched_model
