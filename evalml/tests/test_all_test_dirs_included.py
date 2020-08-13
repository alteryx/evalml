import os

from setuptools import find_packages


def test_all_test_dirs_included():
    all_modules = find_packages()
    test_dir = os.path.dirname(__file__)
    all_test_dirs_with_init_files = [module for module in all_modules if "evalml.tests" in module]
    all_test_dirs = [dirname for dirname, _, files in os.walk(test_dir) if "__pycache__" not in dirname and "test" in os.path.split(dirname)[1]]
    assert len(all_test_dirs) == len(all_test_dirs_with_init_files)
