"""
scripts/run_api.py -- convenience launcher.

Usage: python scripts/run_api.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
