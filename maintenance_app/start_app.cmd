@echo off
cd /d "%~dp0"
set "MAINTAIN_PY=python"
if exist "..\.venv\Scripts\python.exe" set "MAINTAIN_PY=%CD%\..\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "MAINTAIN_PY=%CD%\.venv\Scripts\python.exe"
"%MAINTAIN_PY%" main.py
if errorlevel 1 pause
