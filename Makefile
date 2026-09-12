.PHONY: install demo test lint clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e .

demo:
	$(PYTHON) scripts/run_demo.py --config configs/small_demo.yaml --overwrite

test:
	$(PYTHON) -m pytest -m "not slow"

lint:
	@if $(PYTHON) -c "import ruff" >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check src scripts tests; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check src scripts tests; \
	else \
		echo "ruff is not installed. Install the project (make install) or: pip install ruff"; \
		exit 1; \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
