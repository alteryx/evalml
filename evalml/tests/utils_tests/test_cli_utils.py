import os

import pytest

from evalml.utils import (
    get_evalml_root,
    get_installed_packages,
    get_sys_info,
    print_deps,
    print_info,
    print_sys_info
)

pytest.importorskip('click', reason='Skipping test because click not installed')


@pytest.fixture
def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def test_print_info(capsys):
    print_info()
    out, err = capsys.readouterr()
    assert "EvalML version:" in out
    assert "EvalML installation directory:" in out
    assert "SYSTEM INFO" in out
    assert "INSTALLED VERSIONS" in out
    assert len(err) == 0


def test_print_sys_info(capsys):
    print_sys_info()
    out, err = capsys.readouterr()
    assert "SYSTEM INFO" in out
    assert len(err) == 0


def test_print_deps_info(capsys):
    core_requirements = ["numpy", "pandas", "cloudpickle", "scipy",
                         "scikit-learn", "scikit-optimize", "tqdm", "colorama"]
    print_deps(core_requirements)
    out, err = capsys.readouterr()
    assert "INSTALLED VERSIONS" in out
    for requirement in core_requirements:
        assert requirement in out
    assert len(err) == 0


def test_sys_info():
    sys_info = get_sys_info()
    info_keys = ["python", "python-bits", "OS",
                 "OS-release", "machine", "processor",
                 "byteorder", "LC_ALL", "LANG", "LOCALE"]
    found_keys = [k for k, _ in sys_info]
    assert set(info_keys).issubset(found_keys)


def test_installed_packages():
    installed_packages = get_installed_packages()
    core_requirements = ["numpy", "pandas", "cloudpickle", "scipy",
                         "scikit-learn", "scikit-optimize", "tqdm", "colorama"]
    assert set(core_requirements).issubset(installed_packages.keys())


def test_get_evalml_root(current_dir):
    root = os.path.abspath(os.path.join(current_dir, '..', ".."))
    assert get_evalml_root() == root
