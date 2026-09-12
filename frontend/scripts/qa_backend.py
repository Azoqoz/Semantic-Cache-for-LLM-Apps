"""Real Demo integration server with a disposable database; no production data."""
import sys
import tempfile
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.api import create_app
from src.config import Settings

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="cache-flow-qa-") as directory:
        settings = Settings(app_mode="demo", database_path=Path(directory) / "qa.sqlite3")
        uvicorn.run(create_app(settings=settings), host="127.0.0.1", port=8000, access_log=False)
