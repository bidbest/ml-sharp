@echo off
echo Starting ml-sharp WebUI...
echo.
echo Checking dependencies...
pip install flask -q
pip install -e . -q
echo.
echo Starting server on port 7860 (accessible on local network)
echo Press Ctrl+C to stop the server
echo.
python webui.py --host 0.0.0.0 --preload
