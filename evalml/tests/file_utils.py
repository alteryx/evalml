from pathlib import Path


def get_this_directory() -> Path:
    return Path(__file__).parent.parent.absolute()


def get_datasets_dir() -> str:
    test_input_dir = get_this_directory() / "tests/data"
    return str(test_input_dir)
