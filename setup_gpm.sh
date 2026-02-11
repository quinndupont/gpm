#!/bin/bash
set -e

echo "=== GPM Environment Setup ==="

# Install Ollama
if ! command -v ollama &> /dev/null; then
  echo "Installing Ollama..."
  brew install ollama
else
  echo "Ollama already installed"
fi

# Create Python venv
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/processed data/annotated data/training data/training/gpm_mlx
mkdir -p models/base models/adapters
mkdir -p logs checkpoints
mkdir -p config/prompts

# Pull Ollama model (optional - user can run manually)
echo ""
echo "To pull the annotation model, run:"
echo "  ollama pull llama3.2:3b   # faster, ~2GB"
echo "  # or: ollama pull llama3.2:8b   # better quality, ~5GB"
echo ""
echo "Optimize for Mac Mini M4 (add to ~/.zshrc or ~/.bashrc):"
echo "  export OLLAMA_METAL=1"
echo "  export OLLAMA_NUM_GPU=999"
echo "  export OLLAMA_MAX_LOADED_MODELS=1"
echo "  export OLLAMA_MAX_MEMORY=21474836480"
echo ""
echo "Setup complete. Activate with: source venv/bin/activate"
