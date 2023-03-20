import logging

from keras.engine.sequential import Sequential
from numpy import ndarray

from .arachne import localisation, searchers


def operator_arachne(
    model: Sequential,
    pos: tuple,
    neg: tuple,
    pos_trivial: tuple = None,
    neg_trivial: tuple = None,
) -> None:
    assert model is not None, "Model not found"
    assert len(pos[0]) > 0, "No positive examples"
    assert len(neg[0]) > 0, "No negative examples"
    assert len(neg[0]) >= len(pos[0]), "More positive examples than negative examples"

    pareto, _ = localisation.bidirectional_localisation(model, pos, neg)
    logging.info("Patch localised: %s" % pareto)
    patch_searcher = searchers.DE(model, (pos, neg), (pos_trivial, neg_trivial), pareto)
    patch = patch_searcher.search().x
    patched_model = patch_searcher.apply_patch(patch)
    logging.info("Patch generated: %s" % patch)

    return patched_model
