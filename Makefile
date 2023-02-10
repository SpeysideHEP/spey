.PHONY: all
all:
	make install
	python -m pip install pytest
	make test


.PHONY: install
install:
	pip install -e .


.PHONY: uninstall
uninstall:
	pip uninstall spey

.PHONY: test
test:
	pytest --cov=spey tests/*/*py --cov-fail-under 99


.PHONY: build
build:
	python -m build






