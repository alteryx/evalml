import os

import pytest

from evalml.utils import get_evalml_root, get_installed_packages, get_sys_info


@pytest.fixture
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


def test_sys_info():
    sys_info = get_sys_info()
    info_keys = ["python", "python-bits", "OS",
                 "OS-release", "machine", "processor",
                 "byteorder", "LC_ALL", "LANG", "LOCALE"]
    found_keys = [k for k, _ in sys_info]
    assert set(info_keys).issubset(found_keys)


def test_installed_packages():
    installed_packages = get_installed_packages()
    requirements = ["numpy", "pandas", "tqdm", "toolz", "cloudpickle",
                    "dask", "distributed", "psutil", "Click",
                    "scikit-learn", "pip", "setuptools"]
    assert set(requirements).issubset(installed_packages.keys())


def test_get_featuretools_root(this_dir):
    root = os.path.abspath(os.path.join(this_dir, '..', ".."))
    assert get_evalml_root() == root
