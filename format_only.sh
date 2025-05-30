#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎨 ImageWand Code Formatter${NC}"
echo -e "${BLUE}===========================${NC}"

echo -e "${YELLOW}🔧 Running isort...${NC}"
isort .

echo -e "${YELLOW}🎨 Running black...${NC}"
black .

echo -e "${YELLOW}🔍 Checking formatting...${NC}"

# Verify formatting is correct
isort --check-only --diff . > /dev/null 2>&1
isort_exit=$?

black --check . > /dev/null 2>&1
black_exit=$?

if [ $isort_exit -eq 0 ] && [ $black_exit -eq 0 ]; then
    echo -e "${GREEN}✅ Code is properly formatted!${NC}"
    echo -e "${GREEN}🚀 Ready for CI/CD!${NC}"
else
    echo -e "${YELLOW}⚠️  Some formatting issues may remain. Re-run if needed.${NC}"
fi
