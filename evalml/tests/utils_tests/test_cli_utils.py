import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from evalml.__main__ import cli
from evalml.utils.cli_utils import (
    get_core_requirements,
    get_evalml_root,
    get_installed_packages,
    get_sys_info,
    print_deps,
    print_info,
    print_sys_info
)


@pytest.fixture
def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def test_get_core_requirements():
    assert len(get_core_requirements()) == 13


def test_print_cli_cmd():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_print_cli_info_cmd(caplog):
    runner = CliRunner()
    result = runner.invoke(cli, ['info'])
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


def test_print_deps_info(caplog):
    core_requirements = get_core_requirements()
    print_deps(core_requirements)
    out = caplog.text
    assert "INSTALLED VERSIONS" in out
    for requirement in core_requirements:
        assert requirement in out


def test_sys_info():
    sys_info = get_sys_info()
    info_keys = ["python", "python-bits", "OS",
                 "OS-release", "machine", "processor",
                 "byteorder", "LC_ALL", "LANG", "LOCALE",
                 "# of CPUS", "Available memory"]
    found_keys = [k for k, _ in sys_info]
    assert set(info_keys).issubset(found_keys)


@patch('platform.uname')
def test_sys_info_error(mock_uname):
    mock_uname.side_effects = ValueError()
    assert len(get_sys_info()) == 0
    mock_uname.assert_called()


def test_installed_packages():
    installed_packages = get_installed_packages()
    core_requirements = get_core_requirements()
    assert set(core_requirements).issubset(installed_packages.keys())


def test_get_evalml_root(current_dir):
    root = os.path.abspath(os.path.join(current_dir, '..', ".."))
    assert get_evalml_root() == root
