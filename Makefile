.PHONY: setup setup-extras test eval all

setup:
	pip install -r requirements.txt -r requirements-dev.txt

setup-extras:
	pip install -r requirements-dense.txt -r requirements-gen.txt

test:
	python3 -m pytest tests/ -q

eval:
	./run_all.sh

all: test eval
