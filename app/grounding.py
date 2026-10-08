"""Check that every number in an answer came from a tool result.

Desk Note's rule is that the model never does maths. This is how we check it:
pull the numbers out of the answer and look for each one in the tool results
(or the question). A number stated with less precision is allowed to be a
rounded version of a tool number ("about $14k" for 14,324), but anything that
can't be traced, like two tool figures added together, is reported.
"""

from __future__ import annotations

import json
import re

# $1,234.5k  -2.1%  +3.0pp  12.34  1.2 million
NUMBER = re.compile(
    r"""
    (?<![\w.])                                  # not inside a word, ticker or decimal
    (?P<sign>[-+−])?
    (?P<dollar>\$)?
    (?P<digits>\d{1,3}(?:,\d{3})+|\d+)
    (?P<decimals>\.\d+)?
    (?:\s?(?P<scale>k|K|m|M|bn|thousand|million|billion)\b)?
    (?P<percent>\s?%|\s?pp\b|\s?percentage points?\b)?
    """,
    re.VERBOSE,
)
SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def numbers_in_text(text):
    """[(value, tolerance, original text)] for each number in the text."""
    found = []
    for m in NUMBER.finditer(text):
        if ignorable(text, m):
            continue
        digits = m["digits"].replace(",", "")
        decimals = m["decimals"] or ""
        value = float(digits + decimals)
        scale = SCALES.get((m["scale"] or "").lower(), 1)

        # Tolerance is half the last stated digit: "2.1" means 2.05 to 2.15.
        # Round thousands count as rounded ("$14,000" for 14,324), small round
        # numbers don't ("70%" means 70%).
        if decimals:
            step = 10 ** -(len(decimals) - 1)
        elif value * scale >= 1000 and scale == 1:
            step = 10 ** (len(digits) - len(digits.rstrip("0")))
        else:
            step = 1
        found.append((value * scale, step * scale / 2, m.group(0).strip()))
    return found


def ignorable(text, m):
    """Dates, years, the '500' in S&P 500, and small counts like '3 months'."""
    has_unit = m["dollar"] or m["percent"] or m["scale"] or m["decimals"]
    before = text[max(0, m.start() - 12) : m.start()].lower()
    after = text[m.end() : m.end() + 3]
    if before.endswith(("s&p ", "s&p")) or after.startswith(("-", "/")) and not has_unit:
        return True
    if re.search(MONTHS + r"\s*$", before) or before.endswith(("/", "-")):
        return True
    if has_unit:
        return False
    value = int(m["digits"].replace(",", ""))
    return value <= 31 or 1900 <= value <= 2100


def numbers_in_data(data):
    """Every number in a tool result, including numbers inside its strings."""
    values = []
    if isinstance(data, bool) or data is None:
        return values
    if isinstance(data, int | float):
        values.append(float(data))
    elif isinstance(data, str):
        values.extend(v for v, _, _ in numbers_in_text(data))
    elif isinstance(data, dict):
        for v in data.values():
            values.extend(numbers_in_data(v))
    elif isinstance(data, list | tuple):
        for v in data:
            values.extend(numbers_in_data(v))
    return values


def check(answer, sources):
    """sources: tool result JSON strings and the user's messages.
    Returns {"numbers_checked": n, "ungrounded": [number, ...]}."""
    known = []
    for source in sources:
        try:
            known.extend(numbers_in_data(json.loads(source)))
        except (TypeError, ValueError):
            known.extend(numbers_in_data(str(source)))
    known = [abs(v) for v in known]

    stated = numbers_in_text(answer)
    ungrounded = [
        text
        for value, tolerance, text in stated
        if not any(abs(abs(value) - k) <= tolerance + 1e-9 for k in known)
    ]
    return {"numbers_checked": len(stated), "ungrounded": ungrounded}
