#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 ImageWand Test & Format Script${NC}"
echo -e "${BLUE}=================================${NC}"

# Check if pytest is available
python -m pytest --version &> /dev/null
pytest_available=$?

if [ $pytest_available -ne 0 ]; then
    echo -e "${YELLOW}⚠️  pytest not found in local environment${NC}"
    echo -e "${YELLOW}   This script is designed for CI/CD environments${NC}"
    echo -e "${YELLOW}   Skipping tests and proceeding with formatting...${NC}"
    echo ""

    echo -e "${YELLOW}🔧 Running isort...${NC}"
    isort .

    echo -e "${YELLOW}🎨 Running black...${NC}"
    black .

    echo -e "${GREEN}✅ Code formatted! Ready for CI/CD.${NC}"
    echo -e "${BLUE}💡 To run tests locally, install pytest: pip install pytest${NC}"
    exit 0
fi

echo -e "${YELLOW}🧪 Running tests...${NC}"

# Run tests
python -m pytest tests/ -v

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests passed! Formatting code...${NC}"

    echo -e "${YELLOW}🔧 Running isort...${NC}"
    isort .

    echo -e "${YELLOW}🎨 Running black...${NC}"
    black .

    echo -e "${YELLOW}🔍 Checking formatting...${NC}"

    # Verify formatting is correct
    echo "Checking isort..."
    isort --check-only --diff .
    isort_exit=$?

    echo "Checking black..."
    black --check .
    black_exit=$?

    if [ $isort_exit -eq 0 ] && [ $black_exit -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed and code is properly formatted!${NC}"
        echo -e "${GREEN}✨ Ready for CI/CD! You can safely commit and push.${NC}"
        exit 0
    else
        echo -e "${RED}❌ Formatting issues detected. Please check the output above.${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Tests failed! Please fix the failing tests before formatting.${NC}"
    exit 1
fi
