@echo off
setlocal enabledelayedexpansion

echo Checking Python version...
python --version | findstr "3.13" > nul
if errorlevel 1 (
    echo Error: Python 3.13.x is required but not found.
    echo Please install Python 3.13.x from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip==25.1.1

echo Installing dependencies...
pip install -e .

echo Setup completed successfully!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To start the application, run: python gesture_recognition.py

pause 