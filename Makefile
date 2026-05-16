.PHONY: install test test-cov lint format notebooks clean

install:
	pip install pandas numpy matplotlib seaborn plotly scikit-learn pyarrow pytest pytest-cov ruff jupyter nbconvert

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=voos --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

notebooks:
	jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb

clean:
	rm -rf data/processed/*.parquet __pycache__ .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
