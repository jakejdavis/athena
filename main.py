import importlib
import logging
import os
import pickle
from typing import cast

import click
import keras

import models
import operators.arachne_operators
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


@cli.command(cls=BasicCommand)
@click.argument("subject_name")
@click.argument("test_set")
@click.option(
    "-m",
    "--mutants_dir",
    default="mutants",
    help="Directory to load/save mutated models",
)
@click.option(
    "-r",
    "--metrics_dir",
    default="metrics",
    help="Directory to save the metrics of the mutation tests",
)
def run(subject_name, test_set, mutants_dir, metrics_dir, verbose):
    set_logger_level(verbose)
    logging.info(
        "Running mutation testing on subject %s with test set %s",
        subject_name,
        test_set,
    )
    logging.info("Mutants directory: %s", mutants_dir)
    logging.info("Metrics directory: %s", metrics_dir)

    if not os.path.exists(
        os.path.join(TRAINED_MODELS_DIR, subject_name + "_trained.h5")
    ):
        logging.info(
            f"Trained model for subject {subject_name} not found. Training now..."
        )
        subject_train = importlib.import_module(f"test_models.{subject_name}_train")
        subject_train.main(subject_name)
    logging.info("Loading trained model")
    model = cast(
        keras.Sequential,
        keras.models.load_model(
            os.path.join(TRAINED_MODELS_DIR, subject_name + "_trained.h5")
        ),
    )
    pos, neg = utils.generate_inputs_outputs(subject_name, model)

    patched_model = operators.arachne_operators.operator_arachne(model, pos, neg)
    patched_model.save(os.path.join(TRAINED_MODELS_DIR, subject_name + "_patched.h5"))


@cli.command(cls=BasicCommand)
@click.argument("subject_name")
@click.option(
    "-t",
    "--trained-models-dir",
    default="trained_models",
    help="Directory to load/save trained models",
)
@click.option(
    "-m",
    "--mutants_dir",
    default="mutants",
    help="Directory to load/save mutated models",
)
@click.option(
    "-s",
    "--specific-output",
    default=None,
    help="Specific output to generate mutants for",
)
def generate(subject_name, trained_models_dir, mutants_dir, specific_output, verbose):
    set_logger_level(verbose)
    logging.info(f"Trained models directory: {trained_models_dir}")
    logging.info(f"Mutants directory: {mutants_dir}")

    model_utils = models.get_model(subject_name)()

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

    logging.info("Generating mutant")
    patched_model = operators.arachne_operators.operator_arachne(
        model, pos, neg, pos_trivial=pos_trivial, neg_trivial=neg_trivial
    )

    if not os.path.exists(mutants_dir):
        os.makedirs(mutants_dir)

    patched_model_path = os.path.join(mutants_dir, subject_name + "_patched.h5")
    if specific_output is not None:
        patched_model_path = os.path.join(
            mutants_dir, subject_name + "_patched_" + specific_output + ".h5"
        )
    logging.info(f"Saving mutant to {patched_model_path}")
    patched_model.save(patched_model_path)


@cli.command(cls=BasicCommand)
@click.argument("approach1_mutants_dir")
@click.argument("approach2_mutants_dir")
@click.option(
    "-r",
    "--metrics_dir",
    default="metrics",
    help="Directory to save the metrics of the mutation tests",
)
def compare(approach1_mutants_dir, approach2_mutants_dir, metrics_dir, verbose):
    set_logger_level(verbose)
    logging.info(
        "Comparing mutants from %s and %s", approach1_mutants_dir, approach2_mutants_dir
    )
    logging.info("Metrics directory: %s", metrics_dir)


if __name__ == "__main__":
    cli()
