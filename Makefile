PYTHON ?= python3

.PHONY: figures verify

figures:
	"$(PYTHON)" reproduce.py

verify:
	"$(PYTHON)" -m pytest -q
	"$(PYTHON)" reproduce.py
