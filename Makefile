.PHONY: test format check test-and-format install clean help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -e .
	pip install pytest black isort

test:  ## Run tests
	python -m pytest tests/ -v

format:  ## Format code with black and isort
	isort .
	black .

check:  ## Check code formatting without making changes
	isort --check-only --diff .
	black --check .

test-and-format:  ## Run tests, then format code if tests pass
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v || (echo "❌ Tests failed! Fix tests before formatting." && exit 1)
	@echo "✅ Tests passed! Formatting code..."
	@$(MAKE) format
	@echo "🔍 Verifying formatting..."
	@$(MAKE) check
	@echo "🎉 All tests passed and code is properly formatted!"
	@echo "✨ Ready for CI/CD!"

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
