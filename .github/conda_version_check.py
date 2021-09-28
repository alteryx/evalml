import yaml
from contextlib import contextmanager
import requirements
import pathlib
import os

CONDA_RECIPE = "/Users/freddy.boulton/sources/personal/evalml-core-feedstock/recipe/meta.yaml"
EVALML_PATH = "/Users/freddy.boulton/sources/evalml/"

IGNORE_PACKAGES = {"python", "pmdarima", "pyzmq"}
CONDA_TO_PIP_NAME = {"python-kaleido": "kaleido", 'py-xgboost': 'xgboost', 'matplotlib-base': 'matplotlib',
                     'python-graphviz': 'graphviz'}


@contextmanager
def read_conda_yaml(path):
    with open(path, "rb") as config_file:
        # Toss out the first line that declares the version since its not supported YAML syntax
        next(config_file)
        yield yaml.safe_load(config_file)


def standardize_format(packages):
    standardized_package_specifiers = []
    for package in packages:
        if package.name in IGNORE_PACKAGES:
            continue
        name = CONDA_TO_PIP_NAME.get(package.name, package.name)
        if package.specs:
            all_specs = ",".join([''.join(spec) for spec in package.specs])
            standardized = f"{name}{all_specs}"
        else:
            standardized = name
        standardized_package_specifiers.append(standardized)
    return standardized_package_specifiers


def get_evalml_pip_requirements(evalml_path):
    core_reqs = open(pathlib.Path(evalml_path, "core-requirements.txt")).readlines()
    extra_reqs = open(pathlib.Path(evalml_path, "requirements.txt")).readlines()
    extra_reqs = [req for req in extra_reqs if "-r core-requirements.txt" not in req]
    all_reqs = core_reqs + extra_reqs
    return standardize_format(requirements.parse("".join(all_reqs)))


def get_evalml_conda_requirements(conda_recipe):
    with read_conda_yaml(conda_recipe) as recipe:
        core_reqs = recipe['outputs'][0]['requirements']['run']
        extra_reqs = recipe['outputs'][1]['requirements']['run']
        extra_reqs = [package for package in extra_reqs if "evalml-core" not in package]
        all_reqs = core_reqs + extra_reqs
    return standardize_format(requirements.parse("\n".join(all_reqs)))


def check_versions():
    conda_recipe_file_path = pathlib.Path(os.getcwd(), 'evalml-core-feedstock', 'recipe', 'meta.yaml')
    pip_requirements_path = pathlib.Path(os.getcwd())
    conda_versions = sorted(get_evalml_conda_requirements(conda_recipe_file_path))
    pip_versions = sorted(get_evalml_pip_requirements(pip_requirements_path))
    if conda_versions != pip_versions:
        conda_not_in_pip = set(conda_versions).difference(pip_versions)
        conda_not_in_pip = ["\t" + version for version in conda_not_in_pip]
        conda_not_in_pip = "\n".join(conda_not_in_pip)

        pip_not_in_conda = set(pip_versions).difference(conda_versions)
        pip_not_in_conda = ["\t" + version for version in pip_not_in_conda]
        pip_not_in_conda = "\n".join(pip_not_in_conda)

        raise SystemExit(
            f"The following package versions are different in conda from pip:\n {conda_not_in_pip}\n"
            f"The following package versions are different in pip from conda:\n {pip_not_in_conda}\n"
        )


if __name__ == "__main__":
    check_versions()