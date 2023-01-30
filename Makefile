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
	pytest tests/.


.PHONY: build
build:
	python -m build






