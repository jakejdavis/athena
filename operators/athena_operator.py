import logging

from keras.engine.sequential import Sequential
from numpy import ndarray

from utils.config import get_config_val

from .athena import localisation, searchers
from .athena.searchers.de_multiprocessing import apply_patch
from .operator import Operator


class AthenaOperator(Operator):
    def __init__(self, model, additional_config):
        super().__init__(model, additional_config)

    def __call__(
        self,
        pos: tuple,
        neg: tuple,
        pos_trivial: tuple = None,
        neg_trivial: tuple = None,
    ) -> Sequential:
        assert self.model is not None, "Model not found"

        assert len(pos[0]) > 0, "No positive examples"
        assert len(neg[0]) > 0, "No negative examples"
        assert len(neg[0]) >= len(
            pos[0]
        ), "More positive examples than negative examples"

        localisation_method = get_config_val(
            self.additional_config, "operator.localisation", "bidirectional"
        )
        if localisation_method == "bidirectional":
            logging.info(f"Localising patch using bidirectional localisation")
            patch, _ = localisation.bidirectional_localisation(self.model, pos, neg)
            logging.info("Patch localised: %s" % patch)
        elif localisation_method == "random":
            logging.info(f"Localising patch using random localisation")
            patch = localisation.random_localisation(self.model, pos, neg)
            logging.info("Patch localised: %s" % patch)
        else:
            logging.error(f"Localisation method {localisation_method} not implemented")

        # Set searcher from config
        if (
            get_config_val(self.additional_config, "operator.searcher.name", "de")
            == "de"
        ):
            logging.info(f"Generating patch using differential evolution")
            workers = get_config_val(
                self.additional_config, "operator.searcher.workers", 1, int
            )

            if workers != 1:
                logging.debug(
                    f"Using multiprocessing DE with {'max' if workers == -1 else workers} workers"
                )
                patch_searcher = searchers.DEMultiprocessing(
                    self.model,
                    (pos, neg),
                    (pos_trivial, neg_trivial),
                    patch,
                    self.additional_config,
                    workers,
                )
            else:
                patch_searcher = searchers.DE(
                    self.model,
                    (pos, neg),
                    (pos_trivial, neg_trivial),
                    patch,
                    self.additional_config,
                )

        patched_weights = patch_searcher.search().x
        patched_model = apply_patch(self.model, patched_weights, patch)
        logging.info("Patch generated: %s" % patched_weights)

        return patched_model
