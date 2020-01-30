import click
import pandas as pd
import evalml
from evalml.utils.cli_utils import print_info


@click.group()
def cli():
    pass

@click.command()
def info():
    print_info()

cli.add_command(info)

if __name__ == "__main__":
    cli()
