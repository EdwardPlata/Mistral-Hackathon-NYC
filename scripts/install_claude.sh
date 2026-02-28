#!/bin/bash
set -e

echo "=== Installing Claude Code CLI ==="

# Remove or fix the problematic yarn repository
echo "Fixing apt repositories..."
sudo rm -f /etc/apt/sources.list.d/yarn.list 2>/dev/null || true

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Node.js and npm
echo "Installing Node.js and npm..."
sudo apt-get install -y nodejs npm

# Verify installation
echo "Node version: $(node --version)"
echo "npm version: $(npm --version)"

# Install Claude Code CLI globally
echo "Installing Claude Code CLI..."
sudo npm install -g @anthropic-ai/claude-code

# Verify Claude installation
echo "Verifying Claude installation..."
claude --version

echo "=== Installation complete! ==="
