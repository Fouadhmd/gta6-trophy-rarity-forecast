# GTA VI trophy rarity forecast

An entry for make.fun's `gta6-trophy-rarity-forecast-01` bounty: forecast, for every trophy of Grand Theft Auto VI on
PlayStation 5, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, as five
quantiles a trophy, scored by a pinball loss on the logit scale.

`main.py` reads one case from `$MAKEFUN_INPUT_DIR` (else `/input`), every read the house made of the game's trophy page
before that case's checkpoint, and writes one forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Standard
library only, no network: the grader's sandbox has neither.

## The model

An earn rate is earned / tracked. Both grow while the game is new, the tracked catalogue faster than any trophy's
earners, so every rate but the near-universal ones falls, decelerating as the launch settles. Each trophy's logit rate
is fitted by weighted least squares against ln(1 + hours since release), the weight the players its read tracked, then
shrunken toward the fit's own mean so a rate standing on a handful of people does not carry its own noise into the
slope. A flat page means stale numbers rather than a still game, so a case whose reads never change takes the refresh
offset instead of a zero slope.

## Local checks

    python3 -P -s main.py            # with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set
    python3 dev/simulate.py --runs 24 --horizon 108

`dev/simulate.py` manufactures Exophase-shaped cases offline, quantised to two decimals and whole players the way the
page prints them, and scores this program against the house baseline on the same runs.

MIT licensed.
