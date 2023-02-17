import click
from logger import logger, set_logger_level
import importlib
import os
import h5py
from typing import cast

import operators.arachne_operators
import keras
import utils


TRAINED_MODELS_DIR = "trained_models"

class BasicCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.append(click.Option(['-v', '--verbose'], is_flag=True, help='Enable verbose output'))

@click.group()
def cli():
    pass 

@cli.command(cls=BasicCommand)
@click.argument("subject_name")
@click.argument("test_set")
@click.option("-m", "--mutants_dir", default="mutants", help="Directory to load/save mutated models")
@click.option("-r", "--metrics_dir", default="metrics", help="Directory to save the metrics of the mutation tests")
def run(subject_name, test_set, mutants_dir, metrics_dir, verbose):
    set_logger_level(verbose)
    logger.info("Running mutation testing on subject %s with test set %s", subject_name, test_set)
    logger.info("Mutants directory: %s", mutants_dir)
    logger.info("Metrics directory: %s", metrics_dir)

    if not os.path.exists(os.path.join(TRAINED_MODELS_DIR, subject_name + "_trained.h5")):
        logger.info(f"Trained model for subject {subject_name} not found. Training now...")
        subject_train = importlib.import_module(f"test_models.{subject_name}_train")
        subject_train.main(subject_name)

    #if not os.path.exists(os.path.join(TRAINED_MODELS_DIR, subject_name + "_weak.h5")):
    #    logger.info(f"Weakly trained model for subject {subject_name} not found. Training now...")
    #    subject_train = importlib.import_module(f"test_models.{subject_name}_weak")
    #    subject_train.main(subject_name)

    logger.info("Loading trained model")
    model = cast(keras.Sequential, keras.models.load_model(os.path.join(TRAINED_MODELS_DIR, subject_name + "_trained.h5")))
    i_pos, i_neg = utils.generate_inputs(subject_name, model)    

    operators.arachne_operators.operator_arachne(model, i_pos, i_neg)


@cli.command(cls=BasicCommand)
@click.argument("model_file")
@click.argument("mutants_dir")
def generate(model_file, mutants_dir, verbose):
    set_logger_level(verbose)
    logger.info("Generating mutants for model %s", model_file)
    logger.info("Mutants directory: %s", mutants_dir)

@cli.command(cls=BasicCommand)
@click.argument("approach1_mutants_dir")
@click.argument("approach2_mutants_dir")
@click.option("-r", "--metrics_dir", default="metrics", help="Directory to save the metrics of the mutation tests")
def compare(approach1_mutants_dir, approach2_mutants_dir, metrics_dir, verbose):
    set_logger_level(verbose)
    logger.info("Comparing mutants from %s and %s", approach1_mutants_dir, approach2_mutants_dir)
    logger.info("Metrics directory: %s", metrics_dir)

if __name__ == "__main__":
    cli()
