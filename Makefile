.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	isort --check-only evalml
	sh ./import_or_skip.sh
	python docs/notebook_version_standardizer.py check-versions
	python docs/notebook_version_standardizer.py check-execution
	black evalml -t py39 --check
	pydocstyle evalml/ --convention=google --add-ignore=D107 --add-select=D400 --match-dir='^(?!(tests)).*'
	flake8 evalml

.PHONY: lint-fix
lint-fix:
	black -t py39 evalml
	isort evalml
	python docs/notebook_version_standardizer.py standardize

.PHONY: test
test:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure  --timeout 300

.PHONY: test-no-parallel
test-no-parallel:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure --ignore=evalml/tests/automl_tests/parallel_tests  --timeout 300

.PHONY: test-parallel
test-parallel:
	pytest evalml/tests/automl_tests/parallel_tests/ --timeout 300 --durations 0

.PHONY: doctests
doctests:
	pytest evalml --ignore evalml/tests -n 2 --durations 0 --doctest-modules --doctest-continue-on-failure

.PHONY: git-test-parallel
git-test-parallel:
	pytest evalml/tests/automl_tests/parallel_tests/ -n 1 --cov=evalml --cov-config=pyproject.toml --junitxml=test-reports/git-test-parallel-junit.xml --timeout 300 --durations 0

.PHONY: git-test-automl
git-test-automl:
	pytest evalml/tests/automl_tests evalml/tests/tuner_tests -n 2 --ignore=evalml/tests/automl_tests/parallel_tests --durations 0 --timeout 300 --cov=evalml --cov-config=pyproject.toml --junitxml=test-reports/git-test-automl-junit.xml

.PHONY: git-test-modelunderstanding
git-test-modelunderstanding:
	pytest evalml/tests/model_understanding_tests -n 2 --durations 0 --timeout 300 --cov=evalml --cov-config=pyproject.toml --junitxml=test-reports/git-test-modelunderstanding-junit.xml

.PHONY: git-test-other
git-test-other:
	pytest evalml/tests --ignore evalml/tests/automl_tests/ --ignore evalml/tests/tuner_tests/ --ignore evalml/tests/model_understanding_tests/ --ignore evalml/tests/pipeline_tests/ --ignore evalml/tests/utils_tests/ --ignore evalml/tests/component_tests/test_prophet_regressor.py --ignore evalml/tests/component_tests/test_components.py --ignore evalml/tests/component_tests/test_utils.py --ignore evalml/tests/integration_tests/ -n 2 --durations 0 --timeout 300 --cov=evalml --cov-config=pyproject.toml --junitxml=test-reports/git-test-other-junit.xml
	make doctests

.PHONY: git-test-prophet
git-test-prophet:
	pytest evalml/tests/component_tests/test_prophet_regressor.py evalml/tests/component_tests/test_components.py evalml/tests/component_tests/test_utils.py evalml/tests/pipeline_tests/ evalml/tests/utils_tests/ -n 2 --durations 0 --timeout 300 --cov=evalml --cov-config=pyproject.toml --junitxml=test-reports/git-test-prophet-junit.xml

.PHONY: git-test-integration
git-test-integration:
	pytest evalml/tests/integration_tests -n 2 --durations 0 --timeout 300 --cov=evalml --cov-config=pyproject.toml --junitxml=test-reports/git-test-integration-junit.xml


.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q

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

.PHONY: installdeps-docs
installdeps-docs:
	pip install -e .[docs]

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: package_evalml
package_evalml: upgradepip upgradebuild
	python -m build
	$(eval PACKAGE=$(shell python -c "from pep517.meta import load; metadata = load('.'); print(metadata.version)"))
	tar -zxvf "dist/evalml-${PACKAGE}.tar.gz"
	mv "evalml-${PACKAGE}" unpacked_sdist
