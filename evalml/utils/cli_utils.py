import locale
import os
import platform
import struct
import sys

import pkg_resources
import psutil
import requirements
from psutil._common import bytes2human

import evalml
from evalml.utils import get_logger

logger = get_logger(__file__)


def get_core_requirements():
    reqs_path = os.path.join(os.path.dirname(evalml.__file__), '../core-requirements.txt')
    lines = open(reqs_path, 'r').readlines()
    lines = [line for line in lines if '-r ' not in line]
    reqs = requirements.parse(''.join(lines))
    reqs_names = [req.name for req in reqs]
    return reqs_names


def print_info():
    """Prints information about the system, evalml, and dependencies of evalml.

    Returns:
        None
    """
    logger.log("EvalML version: %s" % evalml.__version__)
    logger.log("EvalML installation directory: %s" % get_evalml_root())
    print_sys_info()
    print_deps(get_core_requirements())


def print_sys_info():
    """Prints system information.

    Returns:
        None
    """
    logger.log("\nSYSTEM INFO")
    logger.log("-----------")
    sys_info = get_sys_info()
    for title, stat in sys_info:
        logger.log("{title}: {stat}".format(title=title, stat=stat))


def print_deps(dependencies):
    """Prints the version number of each dependency.

    Arguments:
        dependencies (list): list of package names to get the version numbers for.

    Returns:
        None
    """
    logger.log("\nINSTALLED VERSIONS")
    logger.log("------------------")
    installed_packages = get_installed_packages()

    packages_to_log = []
    for x in dependencies:
        # prevents uninstalled deps from being printed
        if x in installed_packages:
            packages_to_log.append((x, installed_packages[x]))
    for package, version in packages_to_log:
        logger.log("{package}: {version}".format(package=package, version=version))


# Modified from here
# https://github.com/pandas-dev/pandas/blob/d9a037ec4ad0aab0f5bf2ad18a30554c38299e57/pandas/util/_print_versions.py#L11
def get_sys_info():
    """Returns system information.

    Returns:
        List of tuples about system stats.
    """
    blob = []
    try:
        (sysname, nodename, release,
         version, machine, processor) = platform.uname()
        blob.extend([
            ("python", '.'.join(map(str, sys.version_info))),
            ("python-bits", struct.calcsize("P") * 8),
            ("OS", "{sysname}".format(sysname=sysname)),
            ("OS-release", "{release}".format(release=release)),
            ("machine", "{machine}".format(machine=machine)),
            ("processor", "{processor}".format(processor=processor)),
            ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
            ("LC_ALL", "{lc}".format(lc=os.environ.get('LC_ALL', "None"))),
            ("LANG", "{lang}".format(lang=os.environ.get('LANG', "None"))),
            ("LOCALE", '.'.join(map(str, locale.getlocale()))),
            ("# of CPUS", "{cpus}".format(cpus=psutil.cpu_count())),
            ("Available memory", "{memory}".format(memory=bytes2human(psutil.virtual_memory().available)))
        ])
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
