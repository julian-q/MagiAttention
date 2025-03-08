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

coverage: # FIXME: something is wrong with the 'coverage run', maybe due to multiprocessing
	coverage run -m pytest tests/
	coverage combine
	coverage report
	coverage html

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf dffa/__pycache__
	rm -rf dffa/version.py
	rm -rf build
	rm -rf dist
	rm -rf dffa.egg-info
	rm -rf src/dffa.egg-info
	pip uninstall -y dffa
