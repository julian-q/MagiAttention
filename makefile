.PHONY: refresh build install build_dist json release test clean

refresh: clean build install

build:
	python -m build --wheel

install:
	pip install .

build_dist:
	make clean
	python -m build --wheel
	pip install dist/*.whl
	make test

release:
	python -m twine upload dist/*

test:
	pytest tests/

coverage:
	rm -f .coverage
	rm -f .coverage.*
	coverage run -m pytest tests/
	coverage combine
	coverage report
	coverage html

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf magi_attention/__pycache__
	rm -rf magi_attention/version.py
	rm -rf build
	rm -rf dist
	rm -rf magi_attention.egg-info
	rm -rf src/magi_attention.egg-info
	pip uninstall -y magi_attention
