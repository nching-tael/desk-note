"""Start Desk Note.

    python run.py            # live data from Yahoo Finance
    python run.py --mock     # synthetic data, works offline
"""
import argparse
import logging

import uvicorn
from dotenv import load_dotenv

from app.server import create_app


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Desk Note: your portfolio's morning note, on demand.")
    ap.add_argument("--mock", action="store_true", help="use synthetic market data (no network needed)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"Desk Note running at http://{args.host}:{args.port} ({'mock' if args.mock else 'live'} data)")
    uvicorn.run(create_app(mock=args.mock), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
