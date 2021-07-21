.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	flake8 evalml && isort --check-only evalml && python docs/notebook_version_standardizer.py check-versions
	black evalml --check

.PHONY: lint-fix
lint-fix:
	black evalml
	isort evalml
	python docs/notebook_version_standardizer.py standardize

.PHONY: test
test:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure  --timeout 300

.PHONY: test-no-dask
test-no-dask:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure --ignore=evalml/tests/automl_tests/dask_tests  --timeout 300

.PHONY: test-dask
test-dask:
	pytest evalml/tests/automl_tests/dask_tests/ --doctest-modules --doctest-continue-on-failure  --timeout 300 --durations 0

.PHONY: git-test-dask
git-test-dask:
	pytest evalml/tests/automl_tests/dask_tests/ -n 1 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-dask-junit.xml --doctest-continue-on-failure  --timeout 300 --durations 0

.PHONY: git-test-minimal-deps-dask
git-test-minimal-deps-dask:
	pytest evalml/tests/automl_tests/dask_tests/  -n 1 --doctest-modules --cov=evalml  --junitxml=test-reports/git-test-minimal-deps-dask-junit.xml --doctest-continue-on-failure --has-minimal-dependencies --timeout 300 --durations 0

.PHONY: git-test-automl-core
git-test-automl-core:
	pytest evalml/tests/automl_tests evalml/tests/tuner_tests -n 2 --ignore=evalml/tests/automl_tests/dask_tests --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-automl-core-junit.xml --doctest-continue-on-failure --has-minimal-dependencies

.PHONY: git-test-automl
git-test-automl:
	pytest evalml/tests/automl_tests evalml/tests/tuner_tests -n 2 --ignore=evalml/tests/automl_tests/dask_tests --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-automl-junit.xml --doctest-continue-on-failure

.PHONY: git-test-modelunderstanding-core
git-test-modelunderstanding-core:
	pytest evalml/tests/model_understanding_tests -n 2 --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-modelunderstanding-core-junit.xml --doctest-continue-on-failure --has-minimal-dependencies

.PHONY: git-test-modelunderstanding
git-test-modelunderstanding:
	pytest evalml/tests/model_understanding_tests -n 2 --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-modelunderstanding-junit.xml --doctest-continue-on-failure

.PHONY: git-test-other-core
git-test-other-core:
	pytest evalml/tests --ignore evalml/tests/automl_tests/ --ignore evalml/tests/tuner_tests/ --ignore evalml/tests/model_understanding_tests/ -n 2 --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-other-core-junit.xml --doctest-continue-on-failure --has-minimal-dependencies

.PHONY: git-test-other
git-test-other:
	pytest evalml/tests --ignore evalml/tests/automl_tests/ --ignore evalml/tests/tuner_tests/ --ignore evalml/tests/model_understanding_tests/ -n 2 --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-other-junit.xml --doctest-continue-on-failure

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q

.PHONY: installdeps-core
installdeps-core:
	pip install -e . -q
	pip install -r core-requirements.txt -q

.PHONY: installdeps-test
installdeps-test:
	pip install -e . -q
	pip install -r test-requirements.txt -q

.PHONY: installdeps-dev
installdeps-dev:
	pip install -e . -q
	pip install -r dev-requirements.txt -q

.PHONY: installdeps-docs
installdeps-docs:
	pip install -e . -q
	pip install -r docs-requirements.txt -q
