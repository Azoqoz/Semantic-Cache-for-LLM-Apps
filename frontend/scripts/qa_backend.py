"""Real Demo integration server with a disposable database; no production data."""
import sys
import argparse
import tempfile
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.api import create_app
from src.config import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("demo", "local"), default="demo")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cache-flow-qa-") as directory:
        settings = Settings(app_mode=args.mode, database_path=Path(directory) / "qa.sqlite3")
        uvicorn.run(create_app(settings=settings), host="127.0.0.1", port=args.port, access_log=False)
