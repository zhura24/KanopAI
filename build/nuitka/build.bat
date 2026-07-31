@echo off
REM Quick build script for KanopiAI using Nuitka
REM Usage: Double-click this file or run from command prompt

echo ================================================================================
echo KanopiAI - Nuitka Build Script
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv_build\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv_build
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv_build\Scripts\activate.bat

REM Check if Nuitka is installed
nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Nuitka not found. Installing build requirements...
    python -m pip install --upgrade pip
    pip install -r requirements_build.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
    echo [OK] Build requirements installed
    echo.
)

echo [OK] Nuitka found
nuitka --version
echo.

REM Run build script
echo ================================================================================
echo Starting Build Process...
echo ================================================================================
echo.
echo This will take 15-30 minutes for first build.
echo Please be patient...
echo.

python build_nuitka.py

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo [ERROR] Build failed!
    echo ================================================================================
    echo.
    echo Check the error messages above.
    echo Common issues:
    echo   - MSVC compiler not installed
    echo   - Missing dependencies
    echo   - Out of memory
    echo.
    echo See BUILD_INSTRUCTIONS.md for troubleshooting.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] Build completed!
echo ================================================================================
echo.
echo Executable location: dist\KanopiAI.exe
echo.
echo To test the application:
echo   cd dist
echo   KanopiAI.exe
echo.
pause
