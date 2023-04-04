import click

import commands
import utils.config
import utils.model_utils
import utils.stats
from logger import set_logger_level


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

    commands.run(
        subject_name,
        test_set,
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
    commands.generate(
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

    commands.evaluate(
        subject_name,
        trained_models_dir,
        mutants_dir,
        specific_output,
        additional_config,
    )


if __name__ == "__main__":
    cli()
