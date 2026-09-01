@echo off
chcp 65001 >nul
cd /d "%~dp0"
title VRMA Converter

where python >nul 2>nul
if errorlevel 1 (
    echo [X] ไม่พบ Python - ติดตั้งจาก https://www.python.org/downloads/ ก่อน
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/3] สร้าง virtual environment...
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_ok" (
    echo [2/3] ติดตั้ง dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [X] ติดตั้ง dependencies ไม่สำเร็จ
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps_ok"
)

echo [3/3] เริ่มเซิร์ฟเวอร์...
python run.py
pause