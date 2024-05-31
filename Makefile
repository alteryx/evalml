TIMEOUT ?= 300

.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	sh ./import_or_skip.sh
	python docs/notebook_version_standardizer.py check-versions
	python docs/notebook_version_standardizer.py check-execution
	black . --check --config=./pyproject.toml
	ruff . --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	python docs/notebook_version_standardizer.py standardize
	black . --config=./pyproject.toml
	ruff . --config=./pyproject.toml --fix

.PHONY: test
test:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure  --timeout $(TIMEOUT)

.PHONY: test-no-parallel
test-no-parallel:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure --ignore=evalml/tests/automl_tests/parallel_tests  --timeout $(TIMEOUT)

.PHONY: test-parallel
test-parallel:
	pytest evalml/tests/automl_tests/parallel_tests/ --timeout $(TIMEOUT) --durations 0

.PHONY: doctests
doctests:
	pytest evalml --ignore evalml/tests -n 2 --durations 0 --doctest-modules --doctest-continue-on-failure

.PHONY: git-test-parallel
git-test-parallel:
	pytest evalml/tests/automl_tests/parallel_tests/ -n 1 --cov=evalml --cov-config=./pyproject.toml --junitxml=test-reports/git-test-parallel-junit.xml --timeout $(TIMEOUT) --durations 0

.PHONY: git-test-automl
git-test-automl:
	pytest evalml/tests/automl_tests evalml/tests/tuner_tests -n 2 --ignore=evalml/tests/automl_tests/parallel_tests --durations 0 --timeout $(TIMEOUT) --cov=evalml --cov-config=./pyproject.toml --junitxml=test-reports/git-test-automl-junit.xml

.PHONY: git-test-modelunderstanding
git-test-modelunderstanding:
	pytest evalml/tests/model_understanding_tests -n 2 --durations 0 --timeout $(TIMEOUT) --cov=evalml --cov-config=./pyproject.toml --junitxml=test-reports/git-test-modelunderstanding-junit.xml

.PHONY: git-test-other
git-test-other:
	pytest evalml/tests --ignore evalml/tests/automl_tests/ --ignore evalml/tests/tuner_tests/ --ignore evalml/tests/model_understanding_tests/ --ignore evalml/tests/pipeline_tests/test_pipelines.py --ignore evalml/tests/component_tests/test_prophet_regressor.py --ignore evalml/tests/component_tests/test_components.py --ignore evalml/tests/component_tests/test_utils.py --ignore evalml/tests/integration_tests/ -n 2 --durations 0 --timeout $(TIMEOUT) --cov=evalml --cov-config=./pyproject.toml --junitxml=test-reports/git-test-other-junit.xml
	make doctests

.PHONY: git-test-prophet
git-test-prophet:
	pytest evalml/tests/component_tests/test_prophet_regressor.py evalml/tests/component_tests/test_components.py evalml/tests/component_tests/test_utils.py evalml/tests/pipeline_tests/test_pipelines.py -n 2 --durations 0 --timeout $(TIMEOUT) --cov=evalml --cov-config=./pyproject.toml --junitxml=test-reports/git-test-prophet-junit.xml

.PHONY: git-test-integration
git-test-integration:
	pytest evalml/tests/integration_tests -n 2 --durations 0 --timeout $(TIMEOUT) --cov=evalml --cov-config=./pyproject.toml --junitxml=test-reports/git-test-integration-junit.xml


.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .

.PHONY: installdeps-min
installdeps-min:
	pip install --upgrade pip -q
	pip install -e . --no-dependencies
	pip install -r evalml/tests/dependency_update_check/minimum_test_requirements.txt
	pip install -r evalml/tests/dependency_update_check/minimum_requirements.txt


.PHONY: installdeps-prophet
installdeps-prophet:
	pip install -e .[prophet]

.PHONY: installdeps-test
installdeps-test:
	pip install -e .[test]

.PHONY: installdeps-dev
installdeps-dev:
	pip install -e .[dev]
	pre-commit install

.PHONY: installdeps-docs
installdeps-docs:
	pip install -e .[docs]

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	python -m build
	$(eval PACKAGE=$(shell python -c 'import setuptools; setuptools.setup()' --version))
	tar -zxvf "dist/evalml-${PACKAGE}.tar.gz"
	mv "evalml-${PACKAGE}" unpacked_sdist
