#!/bin/bash

# Script to create a new LLM project from template
# Usage: ./create_project.sh project-name "Project Description"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Check if project name is provided
if [ -z "$1" ]; then
    print_color $RED "Error: Project name is required"
    print_color $YELLOW "Usage: ./create_project.sh project-name \"Project Description\""
    exit 1
fi

PROJECT_NAME="$1"
PROJECT_DESCRIPTION="${2:-A new LLM application}"
WORKSPACE_DIR="$HOME/ai-forge-workspace"
TEMPLATE_DIR="$WORKSPACE_DIR/common/project-templates/llm-app-template"
PROJECT_DIR="$WORKSPACE_DIR/projects/$PROJECT_NAME"

# Check if template exists
if [ ! -d "$TEMPLATE_DIR" ]; then
    print_color $RED "Error: Template directory not found at $TEMPLATE_DIR"
    exit 1
fi

# Check if project already exists
if [ -d "$PROJECT_DIR" ]; then
    print_color $RED "Error: Project '$PROJECT_NAME' already exists"
    exit 1
fi

print_color $BLUE "Creating new LLM project: $PROJECT_NAME"
print_color $YELLOW "Description: $PROJECT_DESCRIPTION"

# Create project directory
mkdir -p "$PROJECT_DIR"

# Copy template files
print_color $BLUE "Copying template files..."
cp -r "$TEMPLATE_DIR"/* "$PROJECT_DIR/"

# Update README.md with project name and description
print_color $BLUE "Updating project files..."
sed -i '' "s/\[PROJECT_NAME\]/$PROJECT_NAME/g" "$PROJECT_DIR/README.md"
sed -i '' "s/Brief description of your LLM application./$PROJECT_DESCRIPTION/g" "$PROJECT_DIR/README.md"

# Update .env.example with project name
sed -i '' "s/Your LLM App/$PROJECT_NAME/g" "$PROJECT_DIR/.env.example"

# Create initial .env file
cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"

# Create additional directories
mkdir -p "$PROJECT_DIR/data/raw"
mkdir -p "$PROJECT_DIR/data/processed"
mkdir -p "$PROJECT_DIR/data/outputs"
mkdir -p "$PROJECT_DIR/models/checkpoints"
mkdir -p "$PROJECT_DIR/models/configs"
mkdir -p "$PROJECT_DIR/models/weights"
mkdir -p "$PROJECT_DIR/tests/unit"
mkdir -p "$PROJECT_DIR/tests/integration"
mkdir -p "$PROJECT_DIR/tests/fixtures"
mkdir -p "$PROJECT_DIR/logs"

# Create __init__.py files
touch "$PROJECT_DIR/src/__init__.py"
touch "$PROJECT_DIR/src/core/__init__.py"
touch "$PROJECT_DIR/src/api/__init__.py"
touch "$PROJECT_DIR/src/utils/__init__.py"
touch "$PROJECT_DIR/src/prompts/__init__.py"
touch "$PROJECT_DIR/src/agents/__init__.py"
touch "$PROJECT_DIR/tests/__init__.py"
touch "$PROJECT_DIR/tests/unit/__init__.py"
touch "$PROJECT_DIR/tests/integration/__init__.py"

# Create a basic test file
cat > "$PROJECT_DIR/tests/test_basic.py" << EOF
"""
Basic tests for $PROJECT_NAME.
"""

import pytest
from src.core.config import get_settings


def test_settings():
    """Test that settings can be loaded."""
    settings = get_settings()
    assert settings.app_name is not None


def test_project_structure():
    """Test that basic project structure exists."""
    import os
    
    # Check that main directories exist
    assert os.path.exists("src")
    assert os.path.exists("tests")
    assert os.path.exists("data")
    assert os.path.exists("config")
EOF

# Initialize git repository
cd "$PROJECT_DIR"
git init
git add .
git commit -m "Initial commit: Create $PROJECT_NAME from template"

print_color $GREEN "✅ Project '$PROJECT_NAME' created successfully!"
print_color $BLUE "Project location: $PROJECT_DIR"
print_color $YELLOW "\nNext steps:"
print_color $YELLOW "1. cd $PROJECT_DIR"
print_color $YELLOW "2. python -m venv venv"
print_color $YELLOW "3. source venv/bin/activate"
print_color $YELLOW "4. pip install -r requirements.txt"
print_color $YELLOW "5. Configure your API keys in .env"
print_color $YELLOW "6. Start developing!"

print_color $GREEN "\nHappy coding! 🚀"
