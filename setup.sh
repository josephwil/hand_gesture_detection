#!/bin/bash

# Check Python version
echo "Checking Python version..."
if ! python3 --version | grep -q "3.13"; then
    echo "Error: Python 3.13.x is required but not found."
    echo "Please install Python 3.13.x from https://www.python.org/downloads/"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip to latest version
echo "Upgrading pip..."
python -m pip install --upgrade pip==25.1.1

# Install dependencies
echo "Installing dependencies..."
pip install -e .

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the application, run: python gesture_recognition.py" 