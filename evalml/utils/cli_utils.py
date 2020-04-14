import locale
import os
import platform
import struct
import sys

import pkg_resources

import evalml
from evalml.utils import Logger

logger = Logger()

core_requirements = ["numpy", "pandas", "cloudpickle", "scipy",
                     "scikit-learn", "scikit-optimize", "tqdm", "colorama"]


def print_info():
    """Prints information about the system, evalml, and dependencies of evalml.

    Returns:
        None
    """
    logger.log("EvalML version: %s" % evalml.__version__)
    logger.log("EvalML installation directory: %s" % get_evalml_root())
    print_sys_info()
    print_deps(core_requirements)


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

    package_dep = []
    for x in dependencies:
        # prevents uninstalled deps from being printed
        if x in installed_packages:
            package_dep.append((x, installed_packages[x]))
    for k, stat in package_dep:
        logger.log("{k}: {stat}".format(k=k, stat=stat))


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
        installed_packages[d.project_name] = d.version
    return installed_packages


def get_evalml_root():
    """Gets location where evalml is installed.

    Returns:
        Location where evalml is installed.
    """
    return os.path.dirname(evalml.__file__)
