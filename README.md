# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fouadhmd/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fouadhmd/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the share of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, as five quantiles per trophy scored by pinball loss on the logit scale.

## Run it

```sh
mkdir -p input out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

## How it forecasts

Each trophy's logit rate is fitted against ln(1 + hours since the first read), weighted by how many earners the rate
rests on, then shrunk onto its tier's pooled slope and read at the target. The spread adds the random walk, the fit's
own residual and the binomial noise of the rate on the players tracked. A page that never changed gets a refresh
offset instead, because a flat page is stale, not still.

## Score it

`dev/simulate.py` manufactures Exophase-shaped cases (a saturating tracked count, refreshes on a 30-minute clock, a
page that stands still between them, rates quantised to 2 decimals) and prints the mean pinball loss of `main.py`
against the house persistence baseline over the runs:

```sh
python3 dev/simulate.py --runs 8 --horizon 6510 --history-hours 66   # the final's shape: 4.5 days, 2.5 days of reads
python3 dev/simulate.py --runs 8 --horizon 90 --history-hours 3 --flat   # the practice's: a flat page, then a refresh
```

| shape | this entry | house baseline |
| --- | --- | --- |
| 6,510-minute horizon | 0.151 | 0.535 |
| 90-minute horizon, flat page | 0.201 | 1.670 |

---

<sub>Entered on [make.fun](https://make.fun/?ref=6sgw00cj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
