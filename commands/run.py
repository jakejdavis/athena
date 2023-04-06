import json
import logging
import os

import keras

import models
import test_sets
import utils.config
import utils.model_utils
import utils.stats

from .generate import generate

TRAINED_MODELS_DIR = "trained_models"
EVALUATION_DIR = "evaluation"
CACHE_DIR = "cache"


def run(
    subject_name,
    test_set,
    trained_models_dir,
    mutants_dir,
    specific_output,
    additional_config,
):
    additional_config = utils.config.load_config(additional_config)

    model_path = os.path.join(trained_models_dir, subject_name + "_trained.h5")
    patched_model_path = os.path.join(
        mutants_dir,
        subject_name
        + "_patched"
        + ("" if specific_output is None else "_" + specific_output)
        + ".h5",
    )
    if os.path.exists(patched_model_path) and os.path.exists(model_path):
        logging.info(f"Loading trained model from {model_path}")
        model = keras.models.load_model(model_path)

        logging.info(f"Loading patched model from {patched_model_path}")
        patched_model = keras.models.load_model(patched_model_path)
    else:
        logging.info("Generating patched model")
        model, patched_model, _ = generate(
            subject_name,
            trained_models_dir,
            mutants_dir,
            specific_output,
            additional_config,
        )

    model_utils = models.get_model(subject_name)(additional_config)

    specific_output_int = None
    if specific_output is not None:
        specific_output_int = int(specific_output)

    inputs, outputs = model_utils.generate_evaluation_data(
        specific_output=specific_output_int, generic=False
    )

    generic_inputs, generic_outputs = model_utils.generate_evaluation_data(
        specific_output=specific_output_int, generic=True
    )

    test_set = test_sets.get_test_set(test_set)(additional_config)

    mutants_killed = 0
    results = {}
    for test_case in test_set.test_cases:
        nonmutated = test_case.run(model, inputs, outputs)
        mutated = test_case.run(patched_model, inputs, outputs)

        mutant_killed = False
        if not test_case.test_passed(nonmutated):
            logging.info(
                f"\u2717 Test case {test_case.name} failed {'with specific outputs' if specific_output is not None else ''} on nonmutated model"
            )
        elif test_case.test_passed(nonmutated) and not test_case.test_passed(mutated):
            logging.info(
                f"\u2713 Mutation killed in test case '{test_case.name}' {'with specific outputs' if specific_output is not None else ''}"
            )
            mutants_killed += 1
            mutant_killed = True
        else:
            logging.info(
                f"\u2717 Mutation not killed in test case '{test_case.name}' {'with specific outputs' if specific_output is not None else ''}"
            )

        if specific_output is not None:
            nonmutated_generic = test_case.run(model, generic_inputs, generic_outputs)
            mutated_generic = test_case.run(
                patched_model, generic_inputs, generic_outputs
            )
            generic_mutant_not_killed = False

            if not test_case.test_passed(nonmutated_generic):
                logging.info(
                    f"\u2717 Test case {test_case.name} with generic outputs failed on nonmutated model"
                )
            elif test_case.test_passed(
                nonmutated_generic
            ) and not test_case.test_passed(mutated_generic):
                logging.info(
                    f"\u2717 Mutation killed in test case '{test_case.name}' with generic outputs"
                )
            else:
                logging.info(
                    f"\u2713 Mutation not killed in test case '{test_case.name}' with generic outputs"
                )
                generic_mutant_not_killed = True

        results[test_case.name] = {
            "nonmutated": nonmutated,
            "mutated": mutated,
            "nonmutated_generic": nonmutated_generic,
            "mutated_generic": mutated_generic,
            "mutant_killed": mutant_killed,
            "generic_mutant_not_killed": generic_mutant_not_killed,
        }

    mutation_score = mutants_killed / len(test_set.test_cases)
    logging.info(f"Mutation score: {mutation_score}/{len(test_set.test_cases)}")
    run_file = utils.config.get_config_val(additional_config, "run.output", "run.json")

    with open(run_file, "w") as f:
        json.dump(
            {
                "mutation_score": mutation_score,
                "test_cases": results,
            },
            f,
            indent=4,
        )
