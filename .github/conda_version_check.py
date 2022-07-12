import os
import pathlib
from contextlib import contextmanager

import requirements
import yaml

from evalml.utils import get_evalml_pip_requirements, standardize_format

IGNORE_PACKAGES = {"python", "pmdarima", "pyzmq", "vowpalwabbit"}


@contextmanager
def read_conda_yaml(path):
    with open(path, "rb") as config_file:
        # Toss out the first line that declares the version since its not supported YAML syntax
        next(config_file)
        yield yaml.safe_load(config_file)


def get_evalml_conda_requirements(conda_recipe):
    with read_conda_yaml(conda_recipe) as recipe:
        core_reqs = recipe["outputs"][0]["requirements"]["run"]
        extra_reqs = recipe["outputs"][1]["requirements"]["run"]
        extra_reqs = [package for package in extra_reqs if "evalml-core" not in package]
        all_reqs = core_reqs + extra_reqs
    return standardize_format(requirements.parse("\n".join(all_reqs)), IGNORE_PACKAGES)


def check_versions():
    conda_recipe_file_path = pathlib.Path(os.getcwd(), ".github", "meta.yaml")
    pip_requirements_path = pathlib.Path(os.getcwd())
    conda_versions = sorted(get_evalml_conda_requirements(conda_recipe_file_path))
    pip_versions = sorted(
        get_evalml_pip_requirements(pip_requirements_path, IGNORE_PACKAGES)
    )
    if conda_versions != pip_versions:
        conda_not_in_pip = set(conda_versions).difference(pip_versions)
        conda_not_in_pip = ["\t" + version for version in conda_not_in_pip]
        conda_not_in_pip = "\n".join(conda_not_in_pip)

        pip_not_in_conda = set(pip_versions).difference(conda_versions)
        pip_not_in_conda = ["\t" + version for version in pip_not_in_conda]
        pip_not_in_conda = "\n".join(pip_not_in_conda)

        raise SystemExit(
            f"The following package versions are different in conda from pip:\n {conda_not_in_pip}\n"
            f"The following package versions are different in pip from conda:\n {pip_not_in_conda}\n",
        )


if __name__ == "__main__":
    check_versions()
