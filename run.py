"""Start the web app.

python run.py           # live prices from Yahoo Finance
python run.py --mock    # synthetic data, works offline
"""

import argparse
import logging

import uvicorn
from dotenv import load_dotenv

from app.server import create_app

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Desk Note web app")
    parser.add_argument("--mock", action="store_true", help="use synthetic market data")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"Desk Note on http://{args.host}:{args.port} ({'mock' if args.mock else 'live'} data)")
    uvicorn.run(create_app(mock=args.mock), host=args.host, port=args.port, log_level="warning")
