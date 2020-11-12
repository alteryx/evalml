import click
import json
import os

DOCS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


def _get_python_version(notebook):
    with open(notebook, "r") as f:
        source = json.load(f)
        version = source['metadata']['language_info']['version']
    return version


def _standardize_python_version(notebook, desired_version='3.7.4'):
    with open(notebook, "r") as f:
        source = json.load(f)
        source['metadata']['language_info']['version'] = desired_version
    json.dump(source, open(notebook, "w"), indent=1)


def _get_ipython_notebooks(docs_source):
    directories_to_skip = ["_templates", "generated", ".ipynb_checkpoints"]
    notebooks = []
    for root, _, filenames in os.walk(docs_source):
        if any(dir_ in root for dir_ in directories_to_skip):
            continue
        for filename in filenames:
            if filename.endswith('.ipynb'):
                notebooks.append(os.path.join(root, filename))
    return notebooks


def _get_notebooks_with_different_versions(notebooks, desired_version='3.7.4'):
    different_versions = []
    for notebook in notebooks:
        version = _get_python_version(notebook)
        if version != desired_version:
            different_versions.append(notebook)
    return different_versions


def _standardize_versions(notebooks, desired_version='3.7.4'):
    for notebook in notebooks:
        _standardize_python_version(notebook, desired_version)


@click.group()
def cli():
    """no-op"""


@cli.command()
@click.option('--desired-version', default='3.7.4', help='python version that all notebooks should match')
def check_versions(desired_version):
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    different_versions = _get_notebooks_with_different_versions(notebooks, desired_version)
    if different_versions:
        different_versions = ['\t' + notebook for notebook in different_versions]
        different_versions = "\n".join(different_versions)
        raise SystemExit(f"The following notebooks don't match {desired_version}:\n {different_versions}")


@cli.command()
@click.option('--desired-version', default='3.7.4', help='python version that all notebooks should match')
def standardize(desired_version):
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    different_versions = _get_notebooks_with_different_versions(notebooks, desired_version)
    if different_versions:
        _standardize_versions(different_versions, desired_version)
        different_versions = ['\t' + notebook for notebook in different_versions]
        different_versions = "\n".join(different_versions)
        click.echo(f"Set the notebook version to {desired_version} for:\n {different_versions}")


if __name__ == '__main__':
    cli()