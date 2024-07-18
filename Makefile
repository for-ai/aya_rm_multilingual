.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := scripts tests

style:
	python -m black --target-version py310 --line-length 119 $(check_dirs) 
	python -m isort $(check_dirs) --profile black -m 9

quality:
	python -m flake8 --max-line-length 119 $(check_dirs)