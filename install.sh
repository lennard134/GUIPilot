#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting installation process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv env

# Activate the virtual environment
echo "Activating the virtual environment..."
source env/bin/activate

# Install dependencies from requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "requirements.txt not found. Skipping Python dependencies installation."
fi

# Check if curl is installed (needed for Ollama installation)
if ! command -v curl &> /dev/null; then
    echo "curl is not installed. Please install curl and try again."
    exit 1
fi

# Install Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | bash
else
    echo "Ollama is already installed."
fi

# Ensure Ollama is accessible
if ! command -v ollama &> /dev/null; then
    echo "Ollama installation failed or it is not in the PATH. Please check your system and try again."
    exit 1
fi

echo "Installation complete. To start the application, activate the virtual environment with:"
echo "source env/bin/activate"
echo "Then run the application with:"
echo "python3 GUI.py"

