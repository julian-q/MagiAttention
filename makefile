.PHONY: refresh build install build_dist json release test clean

refresh: clean build install

build:
	python -m build

install:
	pip install .

build_dist:
	make clean
	python -m build
	pip install dist/*.whl
	make test

release:
	python -m twine upload dist/*

test:
	python -m unittest

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf zeus/__pycache__
	rm -rf zeus/version.py
	rm -rf build
	rm -rf dist
	rm -rf zeus.egg-info
	rm -rf src/zeus.egg-info
	pip uninstall -y zeus
