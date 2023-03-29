import json
import logging
import os
import pickle
import sys

import click
import keras

import models
import operators
import test_sets
import utils
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

    if additional_config is not None:
        if os.path.exists(additional_config):
            with open(additional_config, "r") as f:
                additional_config = json.load(f)
        else:
            logging.debug(
                f"Additional config file {additional_config} not found, trying to load as json"
            )
            additional_config = json.loads(additional_config)
    else:
        additional_config = {}
    logging.info(f"Additional config: {additional_config}")

    # Load/train subject

    model_utils = models.get_model(subject_name)(additional_config)

    trained_model_path = os.path.join(trained_models_dir, subject_name + "_trained.h5")
    if not os.path.exists(trained_model_path):
        logging.info(
            f"Trained model for subject {subject_name} not found. Training now..."
        )
        model = model_utils.train(subject_name)
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

        if not os.path.exists(input_outputs_trivial_path):
            logging.info(
                "Generating trivial inputs and outputs"
                + (" for specific output" if specific_output is not None else "")
            )
            pos_trivial, neg_trivial = model_utils.generate_inputs_outputs(
                model, specific_output=specific_output_int, trivial=True
            )
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

    if not os.path.exists(input_outputs_path):
        logging.info(
            "Generating inputs and outputs"
            + (" for specific output" if specific_output is not None else "")
        )
        pos, neg = model_utils.generate_inputs_outputs(
            model, specific_output=specific_output_int
        )
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
    help="Path to additional configuration json file.",
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

    test_set = test_sets.get_test_set(test_set)(additional_config)
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
            logging.info(f"✓ {test_name} has accuracy {test['value']}")
        else:
            logging.info(f"✗ {test_name} has accuracy {test['value']}")


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
    help="Path to additional configuration json file.",
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
    _generate(
        subject_name,
        trained_models_dir,
        mutants_dir,
        specific_output,
        additional_config,
    )


if __name__ == "__main__":
    cli(prog_name="athena")
