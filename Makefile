.PHONY: setup test eval all

setup:
	pip install -r requirements.txt

test:
	python3 -m pytest tests/ -q

eval:
	./run_all.sh

all: test eval
