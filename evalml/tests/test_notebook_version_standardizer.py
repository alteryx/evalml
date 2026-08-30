import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from docs import notebook_version_standardizer


def test_standardize_removes_only_empty_attachments():
    with tempfile.TemporaryDirectory() as docs_path:
        notebook = Path(docs_path) / "notebook.ipynb"
        notebook.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [],
                            "attachments": {},
                        },
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [],
                            "attachments": {"image.png": {"image/png": "data"}},
                        },
                    ],
                    "metadata": {"language_info": {"version": "3.9.6"}},
                },
            ),
        )
        with patch.object(notebook_version_standardizer, "DOCS_PATH", docs_path):
            result = CliRunner().invoke(
                notebook_version_standardizer.cli,
                ["standardize"],
            )

        assert result.exit_code == 0, result.exception
        cells = json.loads(notebook.read_text())["cells"]
        assert "attachments" not in cells[0]
        assert cells[1]["attachments"] == {"image.png": {"image/png": "data"}}
