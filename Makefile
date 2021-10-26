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
	pytest evalml/tests/automl_tests/parallel_tests/ -n 1 --cov=evalml --junitxml=test-reports/git-test-parallel-junit.xml --timeout 300 --durations 0

.PHONY: git-test-minimal-deps-parallel
git-test-minimal-deps-parallel:
	pytest evalml/tests/automl_tests/parallel_tests/  -n 1 --cov=evalml  --junitxml=test-reports/git-test-minimal-deps-parallel-junit.xml --has-minimal-dependencies --timeout 300 --durations 0

.PHONY: git-test-automl-core
git-test-automl-core:
	pytest evalml/tests/automl_tests evalml/tests/tuner_tests -n 2 --ignore=evalml/tests/automl_tests/parallel_tests --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-automl-core-junit.xml --has-minimal-dependencies

.PHONY: git-test-automl
git-test-automl:
	pytest evalml/tests/automl_tests evalml/tests/tuner_tests -n 2 --ignore=evalml/tests/automl_tests/parallel_tests --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-automl-junit.xml

.PHONY: git-test-modelunderstanding-core
git-test-modelunderstanding-core:
	pytest evalml/tests/model_understanding_tests -n 2 --durations 0 --timeout 300 --cov=evalml --junitxml=test-reports/git-test-modelunderstanding-core-junit.xml --has-minimal-dependencies

.PHONY: git-test-modelunderstanding
git-test-modelunderstanding:
	pytest evalml/tests/model_understanding_tests -n 2 --durations 0 --timeout 300 --cov=evalml --junitxml=test-reports/git-test-modelunderstanding-junit.xml

.PHONY: git-test-other-core
git-test-other-core:
	pytest evalml/tests --ignore evalml/tests/automl_tests/ --ignore evalml/tests/tuner_tests/ --ignore evalml/tests/model_understanding_tests/ -n 2 --durations 0 --cov=evalml --junitxml=test-reports/git-test-other-core-junit.xml --has-minimal-dependencies
	make doctests

.PHONY: git-test-other
git-test-other:
	pytest evalml/tests/component_tests/test_utils.py
	pytest evalml/tests --ignore evalml/tests/automl_tests/ --ignore evalml/tests/tuner_tests/ --ignore evalml/tests/model_understanding_tests/ --ignore evalml/tests/pipeline_tests/ --ignore evalml/tests/utils_tests/ --ignore evalml/tests/component_tests/test_prophet_regressor.py --ignore evalml/tests/component_tests/test_components.py -n 2 --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-other-junit.xml
	make doctests

.PHONY: git-test-prophet
git-test-prophet:
	pytest evalml/tests/component_tests/test_prophet_regressor.py evalml/tests/component_tests/test_components.py evalml/tests/component_tests/test_utils.py evalml/tests/pipeline_tests/ evalml/tests/utils_tests/ -n 2 --durations 0 --timeout 300 --doctest-modules --cov=evalml --junitxml=test-reports/git-test-prophet-junit.xml --doctest-continue-on-failure

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q

SITE_PACKAGES_DIR=$$(python -c 'import site; print(site.getsitepackages()[0])')
.PHONY: installdeps-prophet
installdeps-prophet:
	pip install cmdstanpy==0.9.68
	python ${SITE_PACKAGES_DIR}/cmdstanpy/install_cmdstan.py --dir ${SITE_PACKAGES_DIR} -v 2.28.0
	echo "Installing Prophet with CMDSTANPY backend"
	CMDSTAN=${SITE_PACKAGES_DIR}/cmdstan-2.28.0 STAN_BACKEND=CMDSTANPY pip install --no-cache-dir prophet==1.0.1

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
