#!/bin/bash
# Shell script to run Granite Docling demo on Linux/Mac

echo "Granite Docling 258M - Web Demo Launcher"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check setup
echo "[INFO] Checking setup..."
python3 check_setup.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Setup check failed"
    exit 1
fi

echo ""
echo "[INFO] Starting demo interface..."
echo "[INFO] Open your browser to: http://127.0.0.1:7860"
echo "[INFO] Press Ctrl+C to stop the demo"
echo ""

# Run the simple demo
python3 simple_demo.py

echo ""
echo "[INFO] Demo stopped."