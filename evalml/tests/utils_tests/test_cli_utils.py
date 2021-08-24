import os
import pathlib
from unittest.mock import patch

import pytest
import requirements
from click.testing import CliRunner

from evalml.__main__ import cli
from evalml.utils.cli_utils import (
    get_evalml_root,
    get_installed_packages,
    get_sys_info,
    print_deps,
    print_info,
    print_sys_info,
)


@pytest.fixture
def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_core_requirements(current_dir):
    reqs_path = os.path.join(
        current_dir, pathlib.Path("..", "..", "..", "core-requirements.txt")
    )
    lines = open(reqs_path, "r").readlines()
    lines = [line for line in lines if "-r " not in line]
    reqs = requirements.parse("".join(lines))
    reqs_names = [req.name for req in reqs]
    return reqs_names


def test_print_cli_cmd():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_print_cli_info_cmd(caplog):
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "EvalML version:" in caplog.text
    assert "EvalML installation directory:" in caplog.text
    assert "SYSTEM INFO" in caplog.text
    assert "INSTALLED VERSIONS" in caplog.text


def test_print_info(caplog):
    print_info()
    out = caplog.text
    assert "EvalML version:" in out
    assert "EvalML installation directory:" in out
    assert "SYSTEM INFO" in out
    assert "INSTALLED VERSIONS" in out


def test_print_sys_info(caplog):
    print_sys_info()
    out = caplog.text
    assert "SYSTEM INFO" in out


def test_print_deps_info(caplog, current_dir):
    core_requirements = get_core_requirements(current_dir)
    print_deps()
    out = caplog.text
    assert "INSTALLED VERSIONS" in out
    for requirement in core_requirements:
        assert requirement in out


def test_sys_info():
    sys_info = get_sys_info()
    info_keys = [
        "python",
        "python-bits",
        "OS",
        "OS-release",
        "machine",
        "processor",
        "byteorder",
        "LC_ALL",
        "LANG",
        "LOCALE",
        "# of CPUS",
        "Available memory",
    ]
    found_keys = [k for k, _ in sys_info]
    assert set(info_keys).issubset(found_keys)


@patch("platform.uname")
def test_sys_info_error(mock_uname):
    mock_uname.side_effects = ValueError()
    assert len(get_sys_info()) == 0
    mock_uname.assert_called()


def test_installed_packages(current_dir):
    installed_packages = get_installed_packages()
    core_requirements = get_core_requirements(current_dir)
    assert set(core_requirements).issubset(installed_packages.keys())


def test_get_evalml_root(current_dir):
    root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    assert get_evalml_root() == root
