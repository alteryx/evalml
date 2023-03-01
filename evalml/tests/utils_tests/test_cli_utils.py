import os
from unittest.mock import patch

import black
import pytest
from click.testing import CliRunner
from packaging.requirements import Requirement

from evalml.__main__ import cli
from evalml.utils.cli_utils import (
    get_evalml_black_config,
    get_evalml_pip_requirements,
    get_evalml_requirements_file,
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


def get_requirements(current_dir):
    evalml_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    full_requirements = get_evalml_pip_requirements(evalml_path, convert_to_conda=False)
    reqs = [Requirement(req) for req in full_requirements]
    reqs_names = []
    for req in reqs:
        reqs_names.append(req.name)
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
    requirements = get_requirements(current_dir)
    print_deps()
    out = caplog.text
    assert "INSTALLED VERSIONS" in out
    for requirement in requirements:
        assert requirement in out


def test_get_deps_file(current_dir, tmpdir):
    evalml_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    path = os.path.join(str(tmpdir), "requirements.txt")
    assert not os.path.exists(path)

    requirements_return = get_evalml_requirements_file(evalml_path, path)
    assert os.path.exists(path)

    requirement_file = open(path, "r")
    requirement_file = requirement_file.read()

    requirements = get_requirements(current_dir)
    for requirement in requirements:
        assert requirement in requirement_file

    assert requirement_file == requirements_return


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
    requirements = get_requirements(current_dir)
    assert set(requirements).issubset(installed_packages.keys())


def test_get_evalml_root(current_dir):
    root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    assert get_evalml_root() == root


def test_get_evalml_black_config(current_dir):
    evalml_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    black_config = get_evalml_black_config(evalml_path)
    assert black_config["line_length"] == 88
    assert black_config["target_versions"] == set([black.TargetVersion["PY39"]])

    black_config = get_evalml_black_config(
        os.path.join(current_dir, "..", "..", "random_file"),
    )
    assert black_config["line_length"] == 88
    assert black_config["target_versions"] == set([black.TargetVersion["PY39"]])
