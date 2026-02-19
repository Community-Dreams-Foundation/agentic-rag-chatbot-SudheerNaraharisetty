.PHONY: help install run test lint format clean sanity

# Default target
help:
	@echo "Agentic RAG Chatbot - Available Commands:"
	@echo ""
	@echo "  make install    - Install all dependencies"
	@echo "  make run        - Run the Streamlit application"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run linters (ruff, mypy)"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Clean generated files"
	@echo "  make sanity     - Run sanity check for hackathon"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run Streamlit app
run:
	@echo "Starting Agentic RAG Chatbot..."
	streamlit run app.py

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

# Run linters
lint:
	@echo "Running linters..."
	ruff check src/
	mypy src/ --ignore-missing-imports

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	ruff check --fix src/ tests/

# Clean generated files
clean:
	@echo "Cleaning up..."
	rm -rf data/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -delete 2>/dev/null || true

# Create data directories
data-dirs:
	@mkdir -p data
	@mkdir -p artifacts

# Sanity check for hackathon — generates artifacts/sanity_output.json
# NOTE: sanity_check.sh calls `make sanity`, so we must NOT call sanity_check.sh here
sanity: data-dirs
	@echo "Running sanity check..."
	@python3 scripts/generate_sanity_output.py || python scripts/generate_sanity_output.py
	@echo ""
	@echo "Sanity check complete. Output saved to artifacts/sanity_output.json"

# Development server with auto-reload
dev:
	@echo "Starting development server..."
	streamlit run app.py --server.runOnSave=true

# Setup for first time
setup: install data-dirs
	@echo "Setup complete! Copy .env.example to .env and add your API keys."
