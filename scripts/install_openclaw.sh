#!/bin/bash
set -e

# Suppress UV hardlink warnings in containerized environments
export UV_LINK_MODE=copy

echo "=== Installing OpenClaw ==="

# Install Python package
echo "Installing OpenClaw Python library..."
if command -v uv &> /dev/null; then
    echo "Using uv to install openclaw Python package..."
    uv pip install openclaw
else
    echo "Using pip to install openclaw Python package..."
    pip install openclaw
fi

# Install npm CLI tool
echo ""
echo "Installing OpenClaw CLI tool..."
if command -v npm &> /dev/null; then
    # Check Node.js version
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    echo "Current Node.js version: $(node --version)"
    
    if [ "$NODE_VERSION" -lt 22 ]; then
        echo "OpenClaw requires Node.js 22+. Upgrading Node.js..."
        
        # Install Node.js 22.x using NodeSource
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs
        
        echo "Upgraded to Node.js version: $(node --version)"
    fi
    
    sudo npm install -g openclaw@latest
else
    echo "Warning: npm not found. Skipping CLI installation."
    echo "Run 'bash scripts/install_claude.sh' first to install Node.js/npm."
fi

echo ""
# Verify Python installation
echo "=== Verifying Python Installation ==="
echo "Python location: $(which python)"
echo "Installed packages containing 'claw':"
uv pip list | grep -i claw || echo "No packages with 'claw' found"
echo ""
echo "OpenClaw module info:"
uv run python -c "
import openclaw
print(f'OpenClaw version: {openclaw.__version__}')
print(f'Module location: {openclaw.__file__}')
print('\nAvailable attributes/functions:')
attrs = [attr for attr in dir(openclaw) if not attr.startswith('_')]
for attr in attrs[:20]:  # Show first 20
    print(f'  - {attr}')
if len(attrs) > 20:
    print(f'  ... and {len(attrs) - 20} more')
" 2>&1 || echo "Could not inspect openclaw module"

echo ""
# Verify CLI installation
echo "=== Verifying CLI Installation ==="
if command -v openclaw &> /dev/null; then
    echo "OpenClaw CLI installed at: $(which openclaw)"
    openclaw --version || openclaw --help || echo "OpenClaw CLI available"
else
    echo "OpenClaw CLI not found in PATH"
fi

echo ""
echo "=== OpenClaw installation complete! ==="
echo "Python usage: Import openclaw in your Python scripts with 'import openclaw'"
echo "CLI usage: Run 'openclaw' command in your terminal"
