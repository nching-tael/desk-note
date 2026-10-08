"""Print an attribution and risk report: python -m app.analytics [--mock] [period]"""

import argparse
import json

from .holder import AnalyticsHolder

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("period", nargs="?", default="1w")
parser.add_argument("--mock", action="store_true")
args = parser.parse_args()

analytics = AnalyticsHolder(mock=args.mock).get()
attribution = analytics.attribution(args.period)
print(attribution["headline"], "\n")
for h in attribution["holdings"]:
    print(
        f"  {h['symbol']:<6} total {h['total']:>8,}  market {h['market']:>7,}  "
        f"sector {h['sector_move']:>7,}  stock {h['stock_specific']:>7,}"
    )
print("\nRisk report:")
print(json.dumps(analytics.risk(), indent=2))
