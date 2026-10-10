.PHONY: install test sample train clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

sample:
	python scripts/generate_sample_data.py

train:
	fx-forecast train --data data/sample/exchange_rate_sample.csv --output artifacts

clean:
	rm -rf artifacts .pytest_cache **/__pycache__
