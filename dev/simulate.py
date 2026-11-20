"""Simulate Exophase-shaped cases and score a forecaster against them, offline.

The house grades the final on one case and never shows a practice answer, so the only honest way to compare two
models here is to manufacture the process they forecast: a tracked-player count that grows in refresh steps, and
per-trophy earned counts that grow with the population but at their own difficulty. Reads are quantised to 2 decimals
and to whole players, exactly as the page prints them, so a model that trusts a rate built on four people is punished
the way the real grader punishes it.

One process runs from the launch to the target. The history is the reads only up to the checkpoint; the answer is the
read at the target, from the same growth schedule, so history and truth cannot disagree about the population's pace.

    python3 dev/simulate.py --runs 24 --horizon 6510 --history-hours 66

prints the mean pinball loss of main.py and of the house baseline over the runs, plus the spread of each run's loss.
"""

import argparse
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def pinball(y: float, q: float, tau: float) -> float:
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(answer: dict[int, float], forecast: dict[int, list[float]]) -> float:
    total = 0.0
    items = 0
    for tid, y in answer.items():
        qs = forecast[tid]
        for tau, q in zip(LEVELS, qs):
            total += pinball(y, q, tau)
            items += 1
    return total / items


def build_case(rng: random.Random, index: int, horizon_min: int, history_hours: float, flat: bool = False) -> tuple[dict, datetime]:
    """One synthetic case: a history of reads up to the checkpoint, and the answer read at its target."""
    start = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
    checkpoint = start + timedelta(hours=history_hours, minutes=10 * index)
    target = checkpoint + timedelta(minutes=horizon_min)

    trophies = []
    for tid in range(1, 56):
        tier = "platinum" if tid == 1 else "gold" if tid in (7, 14) else "silver" if tid in (15, 17, 25, 30, 31, 40) else "bronze"
        trophies.append({"id": tid, "name": f"t{tid}", "description": "", "tier": tier, "points": 15, "secret": False})

    # Each trophy's earning hazard: the platinum is rarest, the near-universal opener fastest.
    hazard = {}
    for t in trophies:
        base = {"platinum": 0.0006, "gold": 0.004, "silver": 0.01, "bronze": 0.02}[t["tier"]]
        hazard[t["id"]] = base * rng.lognormvariate(0.0, 0.8)
    hazard[2] = 6.0  # Just Like Paradise, everyone's first

    # A launch state: the first refresh already took the tracked count up, so rates start from real earners.
    cap = rng.uniform(80_000.0, 600_000.0)
    rate_of_growth = rng.uniform(1.1, 1.7)
    tracked = 900.0
    earned = {tid: tracked * hazard[tid] * 0.35 for tid in range(1, 56)}
    earned[2] = tracked * 0.9

    def population(hours: float) -> float:
        """Exophase's tracked count: a saturating catalogue, not a compounding one."""
        return cap / (1.0 + (cap / 900.0 - 1.0) * math.exp(-rate_of_growth * hours))

    history = []
    t = start
    step = timedelta(minutes=30 if flat else 10)
    published = ([round(min(100.0, 100.0 * earned[x] / tracked), 2) for x in range(1, 56)], int(tracked), int(sum(earned.values())))
    jumped = False
    while t <= target:
        hours = (t - start).total_seconds() / 3600.0
        # Exophase refreshes on its own clock, every 30 minutes: the page jumps to the tracked count and every rate
        # is dragged down by the players who arrive holding fewer trophies than the ones already tracked. Between
        # refreshes the page stands still, exactly as it did through the whole launch night, and a page that stands
        # still for hours (the launch night itself) jumps about sixfold once and then stands still again.
        refresh = False
        if flat:
            if t > checkpoint and not jumped:
                tracked *= 6.06
                jumped = True
                refresh = True
        elif t > start and t.minute % 30 == 0:
            tracked = population(hours)
            refresh = True
        if refresh:
            published = (
                [round(min(100.0, 100.0 * earned[x] / tracked), 2) for x in range(1, 56)],
                int(tracked),
                int(sum(earned.values())),
            )
        # earners accrue with the population between refreshes, on the true count, not the shown one
        active = tracked * 0.35
        for tid in earned:
            earned[tid] += active * hazard[tid] / 360.0 * rng.lognormvariate(0.0, 0.15)

        if t <= checkpoint:
            rates, shown_players, shown_earned = published
            history.append(
                {
                    "slot": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "at": t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "players_tracked": shown_players,
                    "trophies_earned": shown_earned,
                    "rates": rates,
                }
            )
        t += step

    answer = {tid: round(min(100.0, 100.0 * earned[tid] / tracked), 2) for tid in earned}

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": f"s{index:02d}",
        "at": checkpoint.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": horizon_min,
        "target": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5", "release_date": "2026-11-19"},
        "trophies": trophies,
        "history": history,
        "notes": "",
    }
    return case, answer


def baseline(case: dict) -> dict[int, list[float]]:
    """The house baseline: the last rate, a random-walk spread."""
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    out = {}
    for i, t in enumerate(case["trophies"]):
        lg = logit(case["history"][-1]["rates"][i])
        out[t["id"]] = [100.0 / (1.0 + math.exp(-(lg + z * spread))) for z in (-1.645, -0.674, 0, 0.674, 1.645)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6510)
    ap.add_argument("--history-hours", type=float, default=66.0)
    ap.add_argument("--flat", action="store_true", help="the page does not change until the checkpoint (launch night)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=str(ROOT / "main.py"))
    args = ap.parse_args()

    model = load(Path(args.model), "candidate")
    rng = random.Random(args.seed)
    mine, house = [], []
    for i in range(args.runs):
        case, answer = build_case(rng, i, args.horizon, args.history_hours, args.flat)
        rows = model.forecast_case(case, case["id"])["trophies"]
        got = {r["id"]: [float(x) for x in r["q"]] for r in rows}
        mine.append(score(answer, got))
        house.append(score(answer, baseline(case)))

    wins = sum(1 for a, b in zip(mine, house) if a < b)
    losses = sum(1 for a, b in zip(mine, house) if a > b)
    mine.sort()
    house.sort()
    report = {
        "runs": args.runs,
        "horizon_minutes": args.horizon,
        "history_hours": args.history_hours,
        "flat": args.flat,
        "model_mean": round(sum(mine) / len(mine), 4),
        "model_best": round(mine[0], 4),
        "model_worst": round(mine[-1], 4),
        "baseline_mean": round(sum(house) / len(house), 4),
        "wins": wins,
        "losses": losses,
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
