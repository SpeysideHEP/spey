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
	python -m setup.py bdist_wheel sdist

.PHONY: check_dist
check_dist:
	twine check dist/*

.PHONY: testpypi
testpypi:
	python3 -m twine upload -r testpypi dist/*

.PHONY: pypi
pypi:
	twine upload dist/*





