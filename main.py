import json
import logging
import os
import pickle
import sys

import click
import keras
import numpy as np
from tqdm import tqdm

import models
import operators
import test_sets
import utils.config
import utils.model_utils
import utils.stats
from logger import set_logger_level

TRAINED_MODELS_DIR = "trained_models"
CACHE_DIR = "cache"


class BasicCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.append(
            click.Option(
                ["-v", "--verbose"], is_flag=True, help="Enable verbose output"
            )
        )


@click.group()
def cli():
    pass


def _generate(
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
        model = model_utils.train(subject_name)

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
    pos_trivial = None
    neg_trivial = None
    if specific_output is not None:
        input_outputs_path = os.path.join(
            CACHE_DIR, subject_name + "_input_outputs_" + specific_output + ".pkl"
        )
        input_outputs_trivial_path = os.path.join(
            CACHE_DIR,
            subject_name + "_input_outputs_trivial_" + specific_output + ".pkl",
        )
        specific_output_int = int(specific_output)

        if not os.path.exists(input_outputs_trivial_path) or not use_cache:
            logging.info(
                "Generating trivial inputs and outputs"
                + (" for specific output" if specific_output is not None else "")
            )
            pos_trivial, neg_trivial = model_utils.generate_inputs_outputs(
                model, specific_output=specific_output_int, trivial=True
            )

            if use_cache:
                pickle.dump(
                    (pos_trivial, neg_trivial), open(input_outputs_trivial_path, "wb")
                )
        else:
            logging.info(
                f"Using cached trivial inputs and outputs from {input_outputs_trivial_path}"
            )
            pos_trivial, neg_trivial = pickle.load(
                open(input_outputs_trivial_path, "rb")
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

    logging.info(f"Generating mutant using {operator_name} operator")
    patched_model = operator(
        pos,
        neg,
        pos_trivial=pos_trivial,
        neg_trivial=neg_trivial,
    )

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

    return model, patched_model


@cli.command(cls=BasicCommand)
@click.argument("subject_name")
@click.argument("test_set")
@click.option(
    "-t",
    "--trained-models-dir",
    default="trained_models",
    help="Directory to load/save trained models.",
)
@click.option(
    "-m",
    "--mutants_dir",
    default="mutants",
    help="Directory to load/save mutated models.",
)
@click.option(
    "-p",
    "--specific-output",
    default=None,
    help="Specific output to generate mutants for.",
)
@click.option(
    "-o",
    "--additional-config",
    help="Path to additional configuration json file or json string.",
)
def run(
    subject_name,
    test_set,
    trained_models_dir,
    mutants_dir,
    specific_output,
    additional_config,
    verbose,
):
    """Runs example test set on subject."""
    set_logger_level(verbose)

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
        model, patched_model = _generate(
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
        specific_output=specific_output_int, trivial=False
    )

    trivial_inputs, trivial_outputs = model_utils.generate_evaluation_data(
        specific_output=specific_output_int, trivial=True
    )

    tests = {
        "Non-mutated model for specific output": {
            "value": test_set.run(model, inputs, outputs),
            "pass": lambda x: test_set.test_passed(x),
        },
        "Non-mutated model excluding specific output": {
            "value": test_set.run(model, trivial_inputs, trivial_outputs),
            "pass": lambda x: test_set.test_passed(x),
        },
        "Mutated model for specific output": {
            "value": test_set.run(patched_model, inputs, outputs),
            "pass": lambda x: not test_set.test_passed(x),
        },
        "Mutated model excluding specific output": {
            "value": test_set.run(patched_model, trivial_inputs, trivial_outputs),
            "pass": lambda x: test_set.test_passed(x),
        },
    }

    for test_name, test in tests.items():
        if test["pass"](test["value"]):
            logging.info(f"\u2713 {test_name} has accuracy {test['value']}")
        else:
            logging.info(f"\u2717 {test_name} has accuracy {test['value']}")


@cli.command(cls=BasicCommand)
@click.argument("subject_name")
@click.option(
    "-t",
    "--trained-models-dir",
    default="trained_models",
    help="Directory to load/save trained models.",
)
@click.option(
    "-m",
    "--mutants_dir",
    default="mutants",
    help="Directory to save mutated models.",
)
@click.option(
    "-p",
    "--specific-output",
    default=None,
    help="Specific output to generate mutants for.",
)
@click.option(
    "-o",
    "--additional-config",
    help="Path to additional configuration json file or json string.",
)
def generate(
    subject_name,
    trained_models_dir,
    mutants_dir,
    specific_output,
    additional_config,
    verbose,
):
    """Generates mutant for subject."""
    set_logger_level(verbose)

    additional_config = utils.config.load_config(additional_config)
    _generate(
        subject_name,
        trained_models_dir,
        mutants_dir,
        specific_output,
        additional_config,
    )


@cli.command(cls=BasicCommand)
@click.argument("subject_name")
@click.option(
    "-t",
    "--trained-models-dir",
    default="trained_models",
    help="Directory to load/save trained models.",
)
@click.option(
    "-m",
    "--mutants_dir",
    default="mutants",
    help="Directory to save mutated models.",
)
@click.option(
    "-p",
    "--specific-output",
    default=None,
    help="Specific output to generate mutants for.",
)
@click.option(
    "-o",
    "--additional-config",
    help="Path to additional configuration json file or json string.",
)
def evaluate(
    subject_name,
    trained_models_dir,
    mutants_dir,
    specific_output,
    additional_config,
    verbose,
):
    """Evaluates a given operator by retraining the model,
    generating a mutant and measuring the effect size of the mutation."""
    set_logger_level(verbose)

    additional_config = utils.config.load_config(additional_config)

    logging.debug(f"Forcing cache to False for evaluation")
    additional_config["cache"] = False

    iterations = utils.config.get_config_val(
        additional_config, "evaluate.iterations", 10, int
    )

    original_effect_sizes = []
    patched_effect_sizes = []

    for i in tqdm(range(iterations)):
        logging.info(f"Running iteration {i+1}/{iterations}")

        model_utils = models.get_model(subject_name)(additional_config)

        if True:
            model, patched_model = _generate(
                subject_name,
                trained_models_dir,
                mutants_dir,
                specific_output,
                additional_config,
            )
        else:
            import tensorflow as tf

            model = tf.keras.models.load_model(
                f"trained_models/{subject_name}_trained.h5"
            )
            patched_model = tf.keras.models.load_model(
                f"mutants/{subject_name}_patched_0.h5"
            )

        # evaluate model and patched_model and calculate effect size
        specific_output_int = None
        if specific_output is not None:
            specific_output_int = int(specific_output)

        inputs, outputs = model_utils.generate_evaluation_data(
            specific_output=specific_output_int, trivial=False
        )

        original_effect_sizes.append(model.evaluate(inputs, outputs, verbose=0)[1])
        patched_effect_sizes.append(
            patched_model.evaluate(inputs, outputs, verbose=0)[1]
        )

        logging.info("Iteration complete!")
        logging.info(f"- Original model accuracy: {original_effect_sizes[-1]}")
        logging.info(f"- Patched model accuracy: {patched_effect_sizes[-1]}")

    original_effect_sizes = np.array(original_effect_sizes)
    patched_effect_sizes = np.array(patched_effect_sizes)

    is_significant, p_value, effect_size = utils.stats.is_significant(
        original_effect_sizes, patched_effect_sizes
    )

    if is_significant:
        logging.info(
            f"\u2713 The effect size of the mutation is significant (p-value = {p_value}, effect size = {effect_size})"
        )
    else:
        logging.info(
            f"\u2717 The effect size of the mutation is not significant (p-value = {p_value}, effect size = {effect_size})"
        )

    # Save evaluation to file
    with open(f"evaluation/{subject_name}_evaluation.json", "w") as f:
        json.dump(
            {
                "original_effect_sizes": original_effect_sizes.tolist(),
                "patched_effect_sizes": patched_effect_sizes.tolist(),
                "p_value": p_value,
                "effect_size": effect_size,
            },
            f,
        )


if __name__ == "__main__":
    cli(prog_name="athena")
