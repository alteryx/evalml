#!/usr/bin/env bash
cd /home/docs/
git clone https://github.com/alteryx/evalml.git
cd evalml/
git checkout --force origin/main
git clean -d -f -f

python3.8 -m virtualenv /home/docs/checkouts/readthedocs.org/user_builds/feature-labs-inc-evalml/envs/main
source /home/docs/checkouts/readthedocs.org/user_builds/feature-labs-inc-evalml/envs/main/bin/activate

python -m pip install --upgrade --no-cache-dir pip
python -m pip install --upgrade --no-cache-dir "setuptools==41.0.1" "docutils==0.14" "mock==1.0.1" "pillow==5.4.1" "alabaster>=0.7,<0.8,!=0.7.5" six "commonmark==0.8.1" "recommonmark==0.5.0" "sphinx<2" "sphinx-rtd-theme<0.5" "readthedocs-sphinx-ext<2.2"
python -m pip install --exists-action=w --no-cache-dir "setuptools>=45.2.0"
python -m pip install --exists-action=w --no-cache-dir -r docs-requirements.txt
python -m pip install --upgrade --upgrade-strategy eager --no-cache-dir .
cd /home/docs/evalml/docs/source/
python /home/docs/checkouts/readthedocs.org/user_builds/feature-labs-inc-evalml/envs/main/bin/sphinx-build -T -E -W --keep-going -d _build/doctrees-readthedocs -D language=en . _build/html
