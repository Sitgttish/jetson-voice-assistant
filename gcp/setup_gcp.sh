#!/bin/bash
set -e

echo "=== GCP VM Setup ==="

# System packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Python venv
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python server.py"
echo ""
echo "NOTE: First run will download the model (~16GB). Set HF_TOKEN if using gated models:"
echo "  export HF_TOKEN=your_token_here"
