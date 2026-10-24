@echo off
title StudentPulse AI
cd /d "%~dp0"
echo ============================================
echo   StudentPulse AI - starting...
echo ============================================
echo.
echo Installing required libraries (first time only, needs internet)...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo Could not install libraries. Check that Python is installed and "Add Python to PATH" was ticked.
    pause
    exit /b 1
)
echo.
echo Opening the app in your browser. Keep this window open. Close it to stop the app.
python -m streamlit run app.py
pause
