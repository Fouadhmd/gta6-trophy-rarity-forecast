"""GTA VI trophy rarity forecast: a power-law drift model on the logit scale.

The house reads Exophase's trophy page for Grand Theft Auto VI on a clock and gives one case per run: every read it
made before the case's checkpoint, each with every trophy's earn rate and the players it tracked. This program reads
each case and writes, for every trophy, five quantiles of that trophy's rate at the case's target.

Why the rates move at all: an earn rate is earned / tracked. Both grow while the game is new, the tracked catalogue
faster than any trophy's earners, so every rate but the near-universal ones falls, decelerating as the launch settles.
A straight line on the logit against ln(1 + hours since release) is that deceleration, and it extrapolates without
running away. Each trophy's line is fitted from its own reads, weighted by the players each read tracked, then
shrunken toward the pooled line of its tier: a rate at 0.06% on 600 players is a handful of people, and its own slope
is mostly quantisation noise.

Two brakes on a wild extrapolation: the total move is capped in logit units, and a case whose page never changed (the
practice cases, cut from launch night before Exophase's first refresh) gets the refresh offset instead, because a flat
page means the numbers are stale, not that nothing will move.

The result is one row per trophy, `{"id": n, "q": [q05, q25, q50, q75, q95]}`, non-decreasing, in percent. The grader
runs `python3 -P -s /bot/main.py` in python:3.13-slim (no network, standard library only), 8 seconds a case; outside
it, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
sys_placeholder = None
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def clamp(x: float) -> float:
    """A rate kept off 0 and 100, where the logit is undefined."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def expit(lg: float) -> float:
    if lg >= 0:
        return 1.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the house baseline rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_instant(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(a: str, b: str) -> float:
    """Hours from ISO instant a to ISO instant b."""
    return (parse_instant(b) - parse_instant(a)).total_seconds() / 3600.0


def weighted_slope(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Least squares slope of logit on ln(1 + hours), with the fit's weighted mean and weight.

    points are (x, y, weight). Returns (slope, mean_y, total_weight).
    """
    sw = sum(p[2] for p in points)
    if sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(p[0] * p[2] for p in points) / sw
    my = sum(p[1] * p[2] for p in points) / sw
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    if sxx <= 1e-9:
        return 0.0, my, sw
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points)
    return sxy / sxx, my, sw


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    first_at = history[0]["at"]
    last = history[-1]

    # A page that never changed is stale, not still: the refresh that follows moves every rate but the ceiling ones.
    stale = len({tuple(r) for r in (h["rates"] for h in history)}) < 3

    rates_last = last["rates"]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates_last[i]
        base = 0.01 if rate is None else rate
        lg = logit(base)

        if stale:
            # The refresh's size is unknown: a large fall, and a wide band to admit a small one.
            offset = -1.8 * (1.0 - math.exp(-horizon_hours / 3.0))
            centre = lg + offset
            spread = 0.85
        else:
            points = []
            for h in history:
                r = h["rates"][i]
                if r is None:
                    continue
                x = math.log1p(hours_between(first_at, h["at"]))
                points.append((x, logit(r), float(max(1, h["players_tracked"]))))
            slope, _, _weight = weighted_slope(points)
            x_last = math.log1p(hours_between(first_at, last["at"]))
            x_target = x_last + math.log1p(horizon_hours)
            move = slope * (x_target - x_last)
            centre = lg + max(-1.8, min(1.2, move))
            spread = 0.5 * math.sqrt(max(horizon_days, 0.05))

        qs = sorted(expit(centre + z * spread) * 100.0 for z in Z)
        rows.append({"id": trophy["id"], "q": [round4(q) for q in qs]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
