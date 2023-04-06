import logging
from typing import Callable

from keras.engine.sequential import Sequential

from utils.config import get_config_val
from utils.model_utils import safe_clone_model

from .athena import localisation, searchers
from .athena.searchers.de import apply_patch
from .operator import Operator


class AthenaOperator(Operator):
    def __init__(self, model, additional_config):
        super().__init__(model, additional_config)

    def __call__(
        self,
        pos: tuple,
        neg: tuple,
        pos_generic: tuple = None,
        neg_generic: tuple = None,
    ) -> Sequential:
        """
        Generate a patch for the model using the given positive and negative examples.

        :param pos: Positive examples
        :param neg: Negative examples
        :param pos_generic: generic positive examples
        :param neg_generic: generic negative examples
        """
        assert self.model is not None, "Model not found"
        assert isinstance(self.model.loss, Callable), "Model loss not callable"

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

        model_copy = safe_clone_model(self.model)

        # Set searcher from config
        if (
            get_config_val(self.additional_config, "operator.searcher.name", "de")
            == "de"
        ):
            logging.info(f"Generating patch using differential evolution")
            workers = get_config_val(
                self.additional_config, "operator.searcher.workers", 1, int
            )

            patch_searcher = searchers.DE(
                model_copy,
                (pos, neg),
                (pos_generic, neg_generic),
                patch,
                self.additional_config,
                workers,
            )

        patched_weights = patch_searcher.search().x
        patched_model = apply_patch(model_copy, patched_weights, patch)
        logging.info("Patch generated: %s" % patched_weights)

        return patched_model
