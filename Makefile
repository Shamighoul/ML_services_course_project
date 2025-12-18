# Define targets
.PHONY: pretty test

pretty: isort black

test_ml:
	pytest tests/test_ml.py

test_app:
	pytest tests/test_app.py

run_app:
	uvicorn app.app:app --host 0.0.0.0 --port 8000