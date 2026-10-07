.PHONY: setup lint typecheck test test-cov demo report app clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest --cov=src/prismrisk --cov-report=term-missing --cov-report=html tests/

demo:
	$(PYTHON) -m prismrisk.cli run --config configs/demo.yaml

report:
	$(PYTHON) -m prismrisk.cli report --config configs/demo.yaml

app:
	streamlit run src/prismrisk/app/streamlit_app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
