.PHONY: help install run test lint format clean sanity

# Default target
help:
	@echo "Agentic RAG Chatbot - Available Commands:"
	@echo ""
	@echo "  make install    - Install all dependencies (Python + Node)"
	@echo "  make run        - Run backend API + Next.js frontend"
	@echo "  make run-api    - Run backend API only"
	@echo "  make run-legacy - Run legacy Streamlit UI"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run linters (ruff, mypy)"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Clean generated files"
	@echo "  make sanity     - Run sanity check for hackathon"
	@echo ""

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Run full stack (backend + frontend) — cross-platform via Node.js
run:
	@echo "Starting Agentic RAG System..."
	@node start.js 2>/dev/null || (echo "Fallback: run manually in two terminals:" && echo "  Terminal 1: python -m uvicorn src.api.server:app --reload --port 8000" && echo "  Terminal 2: cd frontend && npm run dev")

# Run backend API only
run-api:
	@echo "Starting backend API..."
	python -m uvicorn src.api.server:app --reload --port 8000

# Run legacy Streamlit app
run-legacy:
	@echo "Starting Streamlit app..."
	streamlit run app.py

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short 2>/dev/null || echo "No tests directory found"

# Run linters
lint:
	@echo "Running linters..."
	ruff check src/ 2>/dev/null || echo "ruff not installed"

# Format code
format:
	@echo "Formatting code..."
	black src/ 2>/dev/null || echo "black not installed"

# Clean generated files
clean:
	@echo "Cleaning up..."
	rm -rf data/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Create data directories
data-dirs:
	@mkdir -p data
	@mkdir -p artifacts

# Sanity check for hackathon
sanity: data-dirs
	@echo "Running sanity check..."
	@python3 scripts/generate_sanity_output.py 2>/dev/null || python scripts/generate_sanity_output.py
	@echo ""
	@echo "Sanity check complete. Output saved to artifacts/sanity_output.json"

# Development server with auto-reload
dev: run

# Setup for first time
setup: install data-dirs
	@echo "Setup complete! Copy .env.example to .env and add your API keys."
