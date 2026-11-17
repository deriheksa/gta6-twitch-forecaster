# GTA VI Twitch CCU Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/deriheksa/gta6-twitch-forecaster/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/deriheksa/gta6-twitch-forecaster/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (19 November 2026, UTC), from everything recorded
up to the end of the first hour. Submitted to
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
on make.fun.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is the complete standalone entry (Python 3.13 standard library
only, no network, no external files). It reads every `/input/<id>.in` and
writes `/output/<id>.out`: for hours 2–24, five quantiles (5%, 25%, 50%,
75%, 95%) of that hour's concurrent-viewer count. Deterministic, well
under the 8-second per-case limit.

## Method

For a console-only rolling-midnight first-access launch, or any launch
whose hour-1 print exceeds 250,000 viewers:

1. **Trend ridge regression** — per forecast hour, the log ratio of
   viewers to the hour-1 print is regressed on log(hour-1) and the
   pre-checkpoint trend `log(h1 / viewers one hour earlier)`, clipped to
   [-2, 2]. Ridge penalty 0.5, intercept effectively unpenalised. The
   training subset is 25 launch days from 25 games with no prior access
   and hour-1 at least 8% of the day's peak; each game weighs the same
   in total. The log hour-1 predictor is clipped to 0.5 log-units beyond
   the training range, with uncertainty widened in quadrature outside it.
2. **Balanced horizon calibration** — quantile widths are calibrated on
   inner leave-one-game-out folds. The objective weights the console
   cohort and the large-launch cohort equally (per game within each).
   Scale factors are chosen per horizon block (hours 2–6, 7–13, 14–24)
   from [0.65, 0.8, 1, 1.2, 1.5] with a small pull toward 1, and the
   resulting widths are pooled to be non-decreasing over the horizon.

Other targets use a similarity-weighted template of past launch curves
(closeness in log hour-1, unlock regime, console status, prior access).

## Validation (local, on the supplied practice cases)

| Metric | This entry |
| --- | --- |
| Practice, 4 cases (mean pinball loss, log scale) | 0.170054 |
| LOO console first-access, 8 games | 0.141360 |
| LOO large launches (h1 > 250k), 9 days | 0.137635 |

Practice is development data, not a blind test. The entry beats the
house baseline (0.387097) on practice; whether it beats the house AI
model on the unseen final day is unknown.
