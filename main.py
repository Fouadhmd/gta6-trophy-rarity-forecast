"""GTA VI trophy rarity forecast: a hierarchical power-law drift model on the logit scale.

The house reads Exophase's Grand Theft Auto VI trophy page on a clock and gives one case per run: every read before
the case's checkpoint, each with every trophy's earn rate and the players it tracked. This program reads the case and
writes, for every trophy, five quantiles of that trophy's rate at the case's target.

Why the rates move: an earn rate is earned / tracked. The tracked catalogue grows while the game is new, and a new
player holds fewer trophies than the average of the players already tracked, so a refresh drags every rate but the
near-universal ones down. That drag decelerates as the launch settles, which is a line on the logit against
ln(1 + hours since the first read), and read at the target it extrapolates without running away.

Each trophy's line is fitted from its own reads with inverse-variance weights: a rate built on four earners is
quantisation, not signal, so its weight is ~earned and its spread is wide. The per-trophy slope is then shrunk onto
the pooled slope of its tier, because one trophy's few reads cannot carry a slope on their own, and the whole move is
damped, because ln-time extrapolates a deceleration that is itself decaying.

The spread is earned, not guessed: the random walk the house baseline already uses, plus the fit's own weighted
residual, plus the binomial noise of the forecast rate on the players tracked. A case whose page never changed gets
the refresh branch instead, since a flat page is a stale page, not a still one.

The output is one row per trophy, `{"id": n, "q": [q05, q25, q50, q75, q95]}`, non-decreasing, in percent. The
grader runs `python3 -P -s /bot/main.py` in python:3.13-slim (no network, standard library only), 8 seconds a case;
outside it, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The random walk's logit standard deviation per sqrt(day): the house baseline's 0.5, held lower because the fitted
# drift carries the move and a band that wide only spends coverage it does not need.
WALK = 0.28
# How much of the shrunk slope acts over the reach. ln-time is already the damping; 1 is the whole shrunk slope.
DAMP = 1.0
# The pseudo-count of the tier's pooled slope, in reads: a trophy needs this many reads to out-vote its tier.
TIER_PRIOR = 6.0
# A move is capped at this many logits, and at this many per sqrt(day), whichever is larger.
MOVE_CAP = 2.5
MOVE_CAP_PER_ROOT_DAY = 2.0
# Spread is capped so a five-quantile row cannot run away on a trophy nobody has earned.
SPREAD_CAP = 2.6
# A page that has not changed for hours is about to: the drop a refresh makes, how fast its chance accrues, and the
# spread around it, in logits. The launch night jumped 638 to 3,868 players, about 1.8 logits down the page.
STALE_DROP = 2.4
STALE_TAU_HOURS = 1.5
STALE_SPREAD = 0.65


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


def earners(rate: float, players: float) -> float:
    """How many of the tracked players the rate is built on: its evidence, and its noise."""
    return max(0.2, rate / 100.0 * max(1.0, players))


def weight_of(rate: float, players: float) -> float:
    """Inverse variance in logit units: about the earners behind the rate, capped for the near-universal ones."""
    return min(4000.0, earners(rate, players))


def weighted_fit(points: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    """Least squares line of logit on ln(1 + hours), weighted; (slope, intercept, weight, weighted residual)."""
    sw = sum(p[2] for p in points)
    if sw <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    mx = sum(p[0] * p[2] for p in points) / sw
    my = sum(p[1] * p[2] for p in points) / sw
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    if sxx <= 1e-9:
        return 0.0, my, sw, 0.0
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = sum(p[2] * (p[1] - (intercept + slope * p[0])) ** 2 for p in points) / sw
    return slope, intercept, sw, math.sqrt(max(0.0, resid))


def rates_of(reads: list[dict], i: int) -> list[float | None]:
    return [h["rates"][i] for h in reads]


def tier_slopes(reads: list[dict], trophies: list[dict], x_of: list[float]) -> dict[str, float]:
    """The pooled weighted slope of each tier: what a trophy's own slope is pulled toward."""
    acc: dict[str, list[float]] = {}
    for i, trophy in enumerate(trophies):
        points = [
            (x, logit(r), weight_of(r, h["players_tracked"]))
            for x, h, r in zip(x_of, reads, rates_of(reads, i))
            if r is not None
        ]
        slope, _c, sw, _resid = weighted_fit(points)
        if sw <= 0.0:
            continue
        slot = acc.setdefault(trophy.get("tier") or "bronze", [0.0, 0.0])
        slot[0] += slope * sw
        slot[1] += sw
    return {tier: v[0] / v[1] for tier, v in acc.items() if v[1] > 0.0}


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = horizon_hours / 24.0
    first_at = history[0]["at"]
    last = history[-1]
    target_at = case.get("target") or last["at"]
    x_last = math.log1p(max(0.0, hours_between(first_at, last["at"])))
    x_target = math.log1p(max(0.0, hours_between(first_at, target_at)))
    reach = max(0.0, x_target - x_last)

    # A page that never changed is stale, not still: the refresh that follows moves every rate but the ceiling ones.
    stale = len({tuple(r) for r in (h["rates"] for h in history)}) < 3

    x_of = [math.log1p(max(0.0, hours_between(first_at, h["at"]))) for h in history]
    pools = {} if stale else tier_slopes(history, case["trophies"], x_of)

    rates_last = last["rates"]
    players_last = max(1.0, last["players_tracked"])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates_last[i]
        base = 0.01 if rate is None else rate
        lg = logit(base)
        seen = earners(base, players_last)
        p = clamp(base) / 100.0
        # Binomial noise of this rate on the players tracked, in logit units: the floor under every spread.
        noise = 1.0 / math.sqrt(max(0.2, seen * (1.0 - p)))

        if stale:
            # The refresh's size is unknown: a large fall, and a band wide enough to admit a small one.
            offset = -STALE_DROP * (1.0 - math.exp(-horizon_hours / STALE_TAU_HOURS))
            centre = lg + offset
            spread = math.sqrt(WALK**2 * horizon_days + STALE_SPREAD**2)
        else:
            points = [
                (x, logit(r), weight_of(r, h["players_tracked"]))
                for x, h, r in zip(x_of, history, rates_of(history, i))
                if r is not None
            ]
            slope, _intercept, sw, resid = weighted_fit(points)
            if len(points) >= 2 and sw > 0.0:
                prior = pools.get(trophy.get("tier") or "bronze", 0.0)
                n = float(len(points))
                slope = (slope * n + prior * TIER_PRIOR) / (n + TIER_PRIOR)
            move = DAMP * slope * reach
            cap = max(MOVE_CAP, MOVE_CAP_PER_ROOT_DAY * math.sqrt(max(horizon_days, 1e-6)))
            move = max(-cap, min(cap, move))
            centre = lg + move
            spread = math.sqrt(WALK**2 * horizon_days + resid**2 + noise**2)

        spread = min(SPREAD_CAP, max(0.08, spread))
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
