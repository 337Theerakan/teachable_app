@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting app...
streamlit run teachable_app.py

pause