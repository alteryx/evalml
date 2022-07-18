"""CLI functions."""
import configparser
import locale
import os
import pathlib
import platform
import struct
import sys

import pkg_resources
import requirements

import evalml
from evalml.utils import get_logger

CONDA_TO_PIP_NAME = {
    "python-kaleido": "kaleido",
    "py-xgboost": "xgboost",
    "matplotlib-base": "matplotlib",
    "python-graphviz": "graphviz",
}


def print_info():
    """Prints information about the system, evalml, and dependencies of evalml."""
    logger = get_logger(__name__)
    logger.info("EvalML version: %s" % evalml.__version__)
    logger.info("EvalML installation directory: %s" % get_evalml_root())
    print_sys_info()
    print_deps()


def print_sys_info():
    """Prints system information."""
    logger = get_logger(__name__)
    logger.info("\nSYSTEM INFO")
    logger.info("-----------")
    sys_info = get_sys_info()
    for title, stat in sys_info:
        logger.info("{title}: {stat}".format(title=title, stat=stat))


def print_deps():
    """Prints the version number of each dependency."""
    logger = get_logger(__name__)
    logger.info("\nINSTALLED VERSIONS")
    logger.info("------------------")
    installed_packages = get_installed_packages()

    for package, version in installed_packages.items():
        logger.info("{package}: {version}".format(package=package, version=version))


# Modified from here
# https://github.com/pandas-dev/pandas/blob/d9a037ec4ad0aab0f5bf2ad18a30554c38299e57/pandas/util/_print_versions.py#L11
def get_sys_info():
    """Returns system information.

    Returns:
        List of tuples about system stats.
    """
    blob = []
    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", ".".join(map(str, sys.version_info))),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "{sysname}".format(sysname=sysname)),
                ("OS-release", "{release}".format(release=release)),
                ("machine", "{machine}".format(machine=machine)),
                ("processor", "{processor}".format(processor=processor)),
                ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
                ("LC_ALL", "{lc}".format(lc=os.environ.get("LC_ALL", "None"))),
                ("LANG", "{lang}".format(lang=os.environ.get("LANG", "None"))),
                ("LOCALE", ".".join(map(str, locale.getlocale()))),
            ],
        )
    except (KeyError, ValueError):
        pass

    return blob


def get_installed_packages():
    """Get dictionary mapping installed package names to their versions.

    Returns:
        Dictionary mapping installed package names to their versions.
    """
    installed_packages = {}
    for d in pkg_resources.working_set:
        installed_packages[d.project_name.lower()] = d.version
    return installed_packages


def get_evalml_root():
    """Gets location where evalml is installed.

    Returns:
        Location where evalml is installed.
    """
    return os.path.dirname(evalml.__file__)


def standardize_format(packages, ignore_packages=None):
    """Standardizes the format of the given packages.

    Args:
        packages: Requirements package generator object.
        ignore_packages: List of packages to ignore. Defaults to None.

    Returns:
        List of packages with standardized format.
    """
    ignore_packages = [] if ignore_packages is None else ignore_packages
    standardized_package_specifiers = []
    for package in packages:
        if package.name in ignore_packages:
            continue
        name = CONDA_TO_PIP_NAME.get(package.name, package.name)
        if package.specs:
            all_specs = ",".join(["".join(spec) for spec in package.specs])
            standardized = f"{name}{all_specs}"
        else:
            standardized = name
        standardized_package_specifiers.append(standardized)
    return standardized_package_specifiers


def get_evalml_pip_requirements(evalml_path, ignore_packages=None):
    """Gets pip requirements for evalml.

    Args:
        evalml_path: Path to evalml root.
        ignore_packages: List of packages to ignore. Defaults to None.

    Returns:
        List of pip requirements for evalml.
    """
    config = configparser.ConfigParser()
    config.read(pathlib.Path(evalml_path, "setup.cfg"))
    return standardize_format(
        requirements.parse(config["options"]["install_requires"]),
        ignore_packages=ignore_packages,
    )
