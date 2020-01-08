.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 evalml && isort --check-only --recursive evalml

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --select="E225,E222,E303,E261,E241,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" evalml
	isort --recursive evalml

.PHONY: test
test:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure

.PHONY: circleci-test
circleci-test:
	pytest evalml/ -n 8 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure

.PHONY: circleci-test-no-parallel
circleci-test-no-parallel:
	pytest evalml/ --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure

.PHONY: win-circleci-test
win-circleci-test:
	pytest evalml/ -n 4 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q
	pip install -r dev-requirements.txt -q
