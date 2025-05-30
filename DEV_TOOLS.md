# Development Tools

This directory contains automated tools to ensure code quality and CI/CD compliance.

## Quick Start

### 🚀 For CI/CD Compliance
```bash
# Run tests (if available) + format code
./test_and_format.sh
```

### 🎨 Format Code Only
```bash
# Just format the code without tests
./format_only.sh
```

### 📋 Using Makefile
```bash
# See all available commands
make help

# Run tests and format
make test-and-format

# Just format
make format

# Just check formatting (no changes)
make check
```

## Available Scripts

### 1. `test_and_format.sh`
**Recommended for pre-commit workflow**

- ✅ Runs tests (if pytest available)
- 🔧 Formats code with `isort`
- 🎨 Formats code with `black`
- ✨ Verifies formatting compliance
- 🚀 Ensures CI/CD will pass

**Usage:**
```bash
./test_and_format.sh
```

**Features:**
- Smart pytest detection (skips if not available)
- Colorized output
- Fails fast if tests fail
- Only formats if tests pass
- Verifies final formatting

### 2. `format_only.sh`
**Quick formatting without tests**

- 🔧 Runs `isort .`
- 🎨 Runs `black .`
- ✅ Verifies formatting

**Usage:**
```bash
./format_only.sh
```

### 3. `Makefile`
**Professional development workflow**

Available targets:
- `make test` - Run tests only
- `make format` - Format code only
- `make check` - Check formatting without changes
- `make test-and-format` - Full workflow
- `make clean` - Clean up temporary files
- `make install` - Install dependencies

### 4. `.pre-commit-config.yaml`
**Automatic formatting on git commit**

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Now code will be automatically formatted on every commit!

## CI/CD Compliance

These tools ensure your code passes GitHub Actions workflows by:

1. **isort compliance** - Sorts imports correctly
2. **black compliance** - Formats code to PEP 8 standards
3. **Test validation** - Ensures tests pass before formatting

## Workflow Recommendations

### Daily Development
```bash
# Make changes to code
./format_only.sh  # Quick format check
git add .
git commit -m "Your changes"
```

### Before Push/PR
```bash
./test_and_format.sh  # Full validation
git add .
git commit -m "Final formatting"
git push
```

### Using Make (Recommended)
```bash
make test-and-format  # One command for everything
```

### With Pre-commit Hooks (Best)
```bash
git add .
git commit -m "Your changes"  # Auto-formats on commit!
```

## Troubleshooting

### "pytest not found"
```bash
pip install pytest
```

### "isort/black not found"
```bash
pip install black isort
```

### "Tests failing"
Fix the failing tests before formatting:
```bash
python -m pytest tests/ -v  # See detailed failures
```

### "Formatting still wrong"
Re-run the script:
```bash
./format_only.sh  # Sometimes takes 2 runs
```

## Integration with IDEs

### VS Code
Add to `settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "editor.formatOnSave": true
}
```

### PyCharm
1. Install black plugin
2. Configure black as formatter
3. Enable format on save

This ensures your code is always CI/CD ready! 🚀
