#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python) || {
  echo "[X] ไม่พบ Python 3 — ติดตั้งก่อนแล้วรันใหม่"; exit 1; }

if [ ! -d ".venv" ]; then
  echo "[1/3] สร้าง virtual environment..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.deps_ok" ]; then
  echo "[2/3] ติดตั้ง dependencies..."
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  touch .venv/.deps_ok
fi

echo "[3/3] เริ่มเซิร์ฟเวอร์..."
python run.py