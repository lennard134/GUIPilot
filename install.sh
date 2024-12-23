#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting installation process..."

# Check if Python is installed
# Function to check and install the required Python version
check_and_update_python() {
    # Check if Python is installed
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        REQUIRED_VERSION="3.12"

        # Compare Python versions
        if [[ $(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
            echo "Python version is less than $REQUIRED_VERSION. Attempting to update..."
            
            # Update Python to >= 3.12 if necessary
            if [ "$(uname)" == "Darwin" ]; then
                # macOS specific
                brew update
                brew install python@3.12
                brew link --overwrite python@3.12
            elif [ -f /etc/debian_version ]; then
                # Debian/Ubuntu specific
                sudo apt update
                sudo apt install -y python3.12 python3.12-venv python3.12-dev
                sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
                sudo update-alternatives --config python3
            elif [ -f /etc/redhat-release ]; then
                # Red Hat/CentOS specific
                sudo dnf module enable -y python:3.12
                sudo dnf install -y python3.12 python3.12-devel
            else
                echo "Unsupported OS. Please manually update Python to version >= 3.12."
                exit 1
            fi
        else
            echo "Python version is sufficient: $PYTHON_VERSION"
        fi
    else
        echo "Python is not installed. Installing Python 3.12..."
        
        if [ "$(uname)" == "Darwin" ]; then
            # macOS specific
            brew update
            brew install python@3.12
            brew link --overwrite python@3.12
        elif [ -f /etc/debian_version ]; then
            # Debian/Ubuntu specific
            sudo apt update
            sudo apt install -y python3.12 python3.12-venv python3.12-dev
            sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
            sudo update-alternatives --config python3
        elif [ -f /etc/redhat-release ]; then
            # Red Hat/CentOS specific
            sudo dnf module enable -y python:3.12
            sudo dnf install -y python3.12 python3.12-devel
        else
            echo "Unsupported OS. Please manually install Python version >= 3.12."
            exit 1
        fi
    fi
}

# Main script execution
check_and_update_python

# Add the rest of your script logic here
echo "Python is set up. Continuing with the rest of the script..."

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv gui-venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source gui-venv/bin/activate

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

# Pull Ollama models
echo "Pulling Ollama models..."
declare -a models=("llama3:latest" "llama3.2:1b" "llama3.2:latest" "nomic-embed-text:latest")

for model in "${models[@]}"; do
    echo "Pulling model: $model..."
    ollama pull "$model"
    if [ $? -ne 0 ]; then
        echo "Failed to pull model: $model. Please check your connection or model name."
        exit 1
    fi
done

echo "Installation complete. All models have been pulled."
echo "Starting application!" 
echo "Starting it without the bash scripts requires to activate the gui-venv via sourc gui-venv/bin/activate and run python3 GUI.py from the Code directory"
source gui-venv/bin/activate
cd Code
python3 GUI.py
deactivate

# echo "Installation complete. To start the application, activate the virtual environment with:"
# echo "source gui-venv/bin/activate"
# echo "Then run the application with:"
# echo "python3 GUI.py"