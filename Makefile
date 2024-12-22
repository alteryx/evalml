TIMEOUT ?= 300


.PHONY: default
default: install

.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete



.PHONY: test
test:
	uv run pytest evalml/ --doctest-modules --doctest-continue-on-failure  --timeout $(TIMEOUT)


.PHONY: install-uv
install:
	uv sync --extra test --frozen



.PHONY: check
check:
	uv run pre-commit run --all-files
