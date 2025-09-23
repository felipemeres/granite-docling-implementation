@echo off
REM Batch script to run Granite Docling demo on Windows

echo Granite Docling 258M - Web Demo Launcher
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check setup
echo [INFO] Checking setup...
python check_setup.py
if errorlevel 1 (
    echo [ERROR] Setup check failed
    pause
    exit /b 1
)

echo.
echo [INFO] Starting demo interface...
echo [INFO] Open your browser to: http://127.0.0.1:7860
echo [INFO] Press Ctrl+C to stop the demo
echo.

REM Run the simple demo
python simple_demo.py

echo.
echo [INFO] Demo stopped.
pause