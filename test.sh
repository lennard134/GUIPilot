#!/bin/bash

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

