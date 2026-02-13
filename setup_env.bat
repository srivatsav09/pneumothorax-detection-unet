@echo off
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo Virtual environment ready. Activate with: venv\Scripts\activate
