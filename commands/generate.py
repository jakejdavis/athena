import logging
import os
import pickle
import sys
import time

import keras

import models
import operators
import utils.config
import utils.model_utils
import utils.stats

TRAINED_MODELS_DIR = "trained_models"
CACHE_DIR = "cache"


def generate(
    subject_name,
    trained_models_dir,
    mutants_dir,
    specific_output,
    additional_config,
):
    logging.info(f"Trained models directory: {trained_models_dir}")
    logging.info(f"Mutants directory: {mutants_dir}")

    # Load/train subject

    model_utils = models.get_model(subject_name)(additional_config)

    use_cache = utils.config.get_config_val(additional_config, "cache", True, bool)

    trained_model_path = os.path.join(trained_models_dir, subject_name + "_trained.h5")
    if not os.path.exists(trained_model_path) or not use_cache:
        logging.info(
            f"Trained model for subject {subject_name} not found. Training now..."
        )
        model = model_utils.train()

        if not use_cache:
            model.save(trained_model_path)
    else:
        logging.info(f"Loading trained model from {trained_model_path}")
        model = keras.models.load_model(trained_model_path)

    # Generate inputs and outputs for patching

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    input_outputs_path = os.path.join(CACHE_DIR, subject_name + "_input_outputs.pkl")

    specific_output_int = None
    pos_generic = None
    neg_generic = None
    if specific_output is not None:
        input_outputs_path = os.path.join(
            CACHE_DIR, subject_name + "_input_outputs_" + specific_output + ".pkl"
        )
        input_outputs_generic_path = os.path.join(
            CACHE_DIR,
            subject_name + "_input_outputs_generic_" + specific_output + ".pkl",
        )
        specific_output_int = int(specific_output)

        if not os.path.exists(input_outputs_generic_path) or not use_cache:
            logging.info(
                "Generating generic inputs and outputs"
                + (" for specific output" if specific_output is not None else "")
            )
            pos_generic, neg_generic = model_utils.generate_inputs_outputs(
                model, specific_output=specific_output_int, generic=True
            )

            if use_cache:
                pickle.dump(
                    (pos_generic, neg_generic), open(input_outputs_generic_path, "wb")
                )
        else:
            logging.info(
                f"Using cached generic inputs and outputs from {input_outputs_generic_path}"
            )
            pos_generic, neg_generic = pickle.load(
                open(input_outputs_generic_path, "rb")
            )

    if not os.path.exists(input_outputs_path) or not use_cache:
        logging.info(
            "Generating inputs and outputs"
            + (" for specific output" if specific_output is not None else "")
        )
        pos, neg = model_utils.generate_inputs_outputs(
            model, specific_output=specific_output_int
        )

        if use_cache:
            pickle.dump((pos, neg), open(input_outputs_path, "wb"))
    else:
        logging.info(f"Using cached inputs and outputs from {input_outputs_path}")
        pos, neg = pickle.load(open(input_outputs_path, "rb"))

    # Generate mutant from operator

    operator_name = utils.config.get_config_val(
        additional_config, "operator.name", "athena"
    )
    if operator_name is None:
        logging.error("Operator name not found in config")
        sys.exit(1)

    operator = operators.get_operator(operator_name)(model, additional_config)

    time_start = time.time()
    logging.info(f"Generating mutant using {operator_name} operator")
    patched_model = operator(
        pos,
        neg,
        pos_generic=pos_generic,
        neg_generic=neg_generic,
    )
    time_to_generate = time.time() - time_start

    # Save mutant

    if not os.path.exists(mutants_dir):
        os.makedirs(mutants_dir)

    if use_cache:
        patched_model_path = os.path.join(mutants_dir, subject_name + "_patched.h5")
        if specific_output is not None:
            patched_model_path = os.path.join(
                mutants_dir, subject_name + "_patched_" + specific_output + ".h5"
            )
        logging.info(f"Saving mutant to {patched_model_path}")
        patched_model.save(patched_model_path)

    return model, patched_model, time_to_generate
