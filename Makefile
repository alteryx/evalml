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

.PHONY: circleci-test
circleci-test:
	pytest evalml/ -n 8 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure -v

.PHONY: circleci-test-minimal-deps
circleci-test-minimal-deps:
	pytest evalml/ -n 8 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure -v --has-minimal-dependencies

.PHONY: win-circleci-test
win-circleci-test:
	pytest evalml/ -n 8 --doctest-modules --cov=evalml --junitxml=test-reports/junit.xml --doctest-continue-on-failure -v

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install numpy>=1.15.4
	pip install pandas>=1.1.1
	pip install click>=7.1.2
	pip install scikit-learn>=0.21.3
	pip install woodwork==0.0.7 --no-dependencies
	pip install -e .

.PHONY: installdeps-test
installdeps-test:
	pip install --upgrade pip -q
	pip install numpy>=1.15.4
	pip install pandas>=1.1.1
	pip install click>=7.1.2
	pip install scikit-learn>=0.21.3
	pip install woodwork==0.0.7 --no-dependencies
	pip install -e .
	pip install -r test-requirements.txt

.PHONY: installdeps-dev
installdeps-dev:
	pip install --upgrade pip -q
	pip install numpy>=1.15.4
	pip install pandas>=1.1.1
	pip install click>=7.1.2
	pip install scikit-learn>=0.21.3
	pip install woodwork==0.0.7 --no-dependencies
	pip install -e .
	pip install -r dev-requirements.txt

.PHONY: installdeps-docs
installdeps-docs:
	pip install --upgrade pip -q
	pip install numpy>=1.15.4
	pip install pandas>=1.1.1
	pip install click>=7.1.2
	pip install scikit-learn>=0.21.3
	pip install woodwork==0.0.7 --no-dependencies
	pip install -e .
	pip install -r docs-requirements.txt
