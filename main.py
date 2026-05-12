import sys
from pathlib import Path

VENDOR_PATH = Path(__file__).parent / "vendor"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))

from app.main import app, create_app, handler


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
