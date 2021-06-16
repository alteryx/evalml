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
test:
	pytest evalml/tests/automl_tests/dask_tests/ --doctest-modules --doctest-continue-on-failure  --timeout 300

.PHONY: git-test
git-test:
	pytest evalml/ -n 2 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure  --timeout 300

.PHONY: git-test-no-dask
git-test-no-dask:
	pytest evalml/ -n 2 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure  --ignore=evalml/tests/automl_tests/dask_tests --timeout 300

.PHONY: git-test-dask
git-test-dask:
	pytest evalml/tests/automl_tests/dask_tests/ -n 1 --doctest-modules --cov=evalml/tests/automl_tests/dask_tests/ --junitxml=test-reports/junit.xml --doctest-continue-on-failure  --timeout 300

.PHONY: git-test-nocov
git-test-nocov:
	pytest evalml/ -n 2 --doctest-modules --doctest-continue-on-failure  --ignore=evalml/tests/automl_tests/dask_tests --timeout 300

.PHONY: git-test-minimal-deps
git-test-minimal-deps:
	pytest evalml/ -n 2 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure --has-minimal-dependencies --timeout 300

.PHONY: git-test-minimal-deps-no-dask
git-test-minimal-deps-no-dask:
	pytest evalml/ -n 2 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure --has-minimal-dependencies  --ignore=evalml/tests/automl_tests/dask_tests --timeout 300

.PHONY: git-test-minimal-deps-dask
git-test-minimal-deps-dask:
	pytest evalml/tests/automl_tests/dask_tests/  -n 1 --doctest-modules --cov=evalml/tests/automl_tests/dask_tests/  --junitxml=test-reports/junit.xml --doctest-continue-on-failure --has-minimal-dependencies

.PHONY: git-test-minimal-deps-nocov
git-test-minimal-deps-nocov:
	pytest evalml/ -n 2 --doctest-modules --doctest-continue-on-failure --has-minimal-dependencies  --ignore=evalml/tests/automl_tests/dask_tests --timeout 300

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
