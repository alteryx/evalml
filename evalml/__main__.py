"""I'm a docstring."""

import click

from evalml.utils.cli_utils import print_info


@click.group()
def cli():
    """I'm a docstring."""
    pass


@click.command()
def info():
    """I'm a docstring."""
    print_info()


cli.add_command(info)
