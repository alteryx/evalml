.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	flake8 evalml && isort --check-only --recursive evalml && python docs/notebook_version_standardizer.py check-versions

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --select="E225,E222,E303,E261,E241,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" evalml
	isort --recursive evalml
	python docs/notebook_version_standardizer.py standardize

.PHONY: test
test:
	pytest evalml/ --doctest-modules --doctest-continue-on-failure

.PHONY: git-test
git-test:
	pytest evalml/ -n 4 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure

.PHONY: git-test-minimal-deps
git-test-minimal-deps:
	pytest evalml/ -n 4 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure --has-minimal-dependencies

.PHONY: win-git-test
win-git-test:
	pytest evalml/ -n 4 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure

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
