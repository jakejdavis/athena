import json
import logging
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import models
import utils.config
import utils.model_utils
import utils.stats

from .generate import generate

TRAINED_MODELS_DIR = "trained_models"
EVALUATION_DIR = "evaluation"
CACHE_DIR = "cache"


def evaluate(
    subject_name,
    trained_models_dir,
    mutants_dir,
    specific_output,
    additional_config,
):
    cache = utils.config.get_config_val(additional_config, "cache", True, bool)

    iterations = utils.config.get_config_val(
        additional_config, "evaluate.iterations", 10, int
    )

    original_model_accuracy = []
    patched_model_accuracy = []

    if specific_output is not None:
        original_model_accuracy_generic = []
        patched_model_accuracy_generic = []

    times_to_generate = []

    for i in tqdm(range(iterations)):
        logging.info(f"Running iteration {i+1}/{iterations}")

        model_path = os.path.join(
            EVALUATION_DIR,
            "models",
            "original",
            f"{subject_name}{'_' + specific_output if specific_output is not None else ''}.h5",
        )
        patched_model_path = os.path.join(
            EVALUATION_DIR,
            "models",
            "mutants",
            f"{subject_name}{'_' + specific_output if specific_output is not None else ''}_{i}.h5",
        )

        model_utils = models.get_model(subject_name)(additional_config)

        if os.path.exists(model_path) and os.path.exists(patched_model_path) and cache:
            logging.info(f"Loading models from disk")
            model = tf.keras.models.load_model(model_path)
            patched_model = tf.keras.models.load_model(patched_model_path)
            time_to_generate = 0
        else:
            logging.debug(f"Forcing cache to False for evaluation mutant generation")
            additional_config["cache"] = False
            model, patched_model, time_to_generate = generate(
                subject_name,
                trained_models_dir,
                mutants_dir,
                specific_output,
                additional_config,
            )

            logging.info(f"Saving models to disk")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(patched_model_path), exist_ok=True)

            model.save(model_path)
            patched_model.save(patched_model_path)

        times_to_generate.append(time_to_generate)

        # Evaluate model and patched_model and calculate effect size
        specific_output_int = None
        if specific_output is not None:
            specific_output_int = int(specific_output)

        inputs, outputs = model_utils.generate_evaluation_data(
            specific_output=specific_output_int, generic=False
        )

        original_model_accuracy.append(model.evaluate(inputs, outputs, verbose=0)[1])
        patched_model_accuracy.append(
            patched_model.evaluate(inputs, outputs, verbose=0)[1]
        )

        generic_inputs, generic_outputs = model_utils.generate_evaluation_data(
            specific_output=specific_output_int, generic=True
        )

        logging.info(f"Iteration complete (took {times_to_generate[-1]} seconds)")
        logging.info(f"- Original model accuracy: {original_model_accuracy[-1]}")
        logging.info(f"- Patched model accuracy: {patched_model_accuracy[-1]}")

        if specific_output is not None:
            original_model_accuracy_generic.append(
                model.evaluate(generic_inputs, generic_outputs, verbose=0)[1]
            )
            patched_model_accuracy_generic.append(
                patched_model.evaluate(generic_inputs, generic_outputs, verbose=0)[1]
            )
            logging.info(
                f"- Original model accuracy (generic): {original_model_accuracy_generic[-1]}"
            )
            logging.info(
                f"- Patched model accuracy (generic): {patched_model_accuracy_generic[-1]}"
            )

    original_model_accuracy = np.array(original_model_accuracy)
    patched_model_accuracy = np.array(patched_model_accuracy)

    if specific_output is not None:
        original_model_accuracy_generic = np.array(original_model_accuracy_generic)
        patched_model_accuracy_generic = np.array(patched_model_accuracy_generic)

    try:
        is_significant, p_value, effect_size = utils.stats.is_significant(
            original_model_accuracy, patched_model_accuracy
        )

        if is_significant:
            logging.info(
                f"\u2713 The effect size of the mutation is significant (p-value = {p_value}, effect size = {effect_size})"
            )
        else:
            logging.info(
                f"\u2717 The effect size of the mutation is not significant (p-value = {p_value}, effect size = {effect_size})"
            )
    except Exception as e:
        logging.error(f"Could not calculate effect size: {e}")
        p_value = None
        effect_size = None

    # Save evaluation to file
    with open(
        f"evaluation/{subject_name}_evaluation{'_' + specific_output if specific_output is not None else ''}.json",
        "w",
    ) as f:
        json.dump(
            {
                "original_model_accuracy": original_model_accuracy.tolist(),
                "patched_model_accuracy": patched_model_accuracy.tolist(),
                "original_model_accuracy_generic": original_model_accuracy_generic.tolist()
                if specific_output is not None
                else None,
                "patched_model_accuracy_generic": patched_model_accuracy_generic.tolist()
                if specific_output is not None
                else None,
                "time_to_generate": times_to_generate,
                "p_value": p_value,
                "effect_size": effect_size,
            },
            f,
        )
