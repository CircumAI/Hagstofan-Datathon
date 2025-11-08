#!/bin/bash
# Startup script for Iceland Export Prediction Dashboard

echo "========================================="
echo "Iceland Export Prediction Dashboard"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements-web.txt

# Start the Flask app
echo ""
echo "Starting dashboard..."
echo "========================================="
echo "Dashboard will be available at:"
echo "  http://localhost:5000"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
