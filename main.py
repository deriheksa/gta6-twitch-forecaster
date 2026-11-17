"""GTA VI Twitch launch-day forecaster.

Reads one case per file from /input (or MAKEFUN_INPUT_DIR) and writes, for
each of hours 2..24 of the release day, five quantiles (5/25/50/75/95%) of
that hour's concurrent-viewer count to /output (or MAKEFUN_OUTPUT_DIR).

Method: for a console-only rolling-midnight first-access launch, or any
launch whose hour-1 print exceeds 250,000 viewers, fit a ridge regression
per forecast hour on the log ratio of viewers to the hour-1 print, using
log(hour-1) and the pre-checkpoint trend as predictors, then calibrate the
quantile widths on inner leave-one-game-out folds (balanced between the
console cohort and the large-launch cohort, three horizon blocks). Other
targets use a similarity-weighted template of past launch curves.

Python 3.13 standard library only. No network, no external files, no
practice truth, no case-id lookup, no target data after hour 1.
"""
import json
import math
import os
import re
import statistics as st
from datetime import datetime, timedelta
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZVALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = tuple(range(2, 25))

# First-access calendar days documented independently of the kit's official
# release days. A global opening shortly before UTC midnight is recorded on
# the following UTC day, matching how the supplied release-day examples are
# represented.
EXTRA_DATES = {
    'Hogwarts Legacy': ('2023-02-07', 'mixed'),
    'Diablo IV': ('2023-06-02', 'global'),
    'Starfield': ('2023-09-01', 'global'),
    'Death Stranding 2: On the Beach': ('2025-06-24', 'rolling'),
    'Saros': ('2026-04-28', 'rolling'),
    'Forza Horizon 5': ('2021-11-05', 'mixed'),
    'Forza Horizon 6': ('2026-05-15', 'mixed'),
    'DOOM: The Dark Ages': ('2025-05-13', 'global'),
}


def _clip(value, low, high):
    return min(high, max(low, value))


def _log(value):
    """Natural log of a viewer count, floored at 1 so zero counts are safe."""
    return math.log(max(1.0, float(value)))


def _quantile(sorted_values, q):
    """Linear-interpolation quantile of an already sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    pos = (n - 1) * q
    i = int(pos)
    nxt = min(i + 1, n - 1)
    return sorted_values[i] + (sorted_values[nxt] - sorted_values[i]) * (pos - i)


def _weighted_quantile(values, weights, q):
    """Smallest value whose cumulative weight reaches q of the total."""
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    target = q * total
    seen = 0.0
    for value, weight in pairs:
        seen += weight
        if seen >= target:
            return value
    return pairs[-1][0]


def _solve(matrix, rhs):
    """Dense linear solve by Gauss-Jordan elimination with partial pivoting."""
    n = len(rhs)
    rows = [list(row) + [v] for row, v in zip(matrix, rhs)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(rows[i][col]))
        rows[col], rows[piv] = rows[piv], rows[col]
        den = rows[col][col]
        if abs(den) < 1e-10:
            den = 1e-10
        for k in range(col, n + 1):
            rows[col][k] /= den
        for i in range(n):
            if i == col:
                continue
            w = rows[i][col]
            for k in range(col, n + 1):
                rows[i][k] -= w * rows[col][k]
    return [rows[i][-1] for i in range(n)]


def _pava(values):
    """Pool-adjacent-violators: smallest non-decreasing sequence above input.

    Adjacent blocks are merged while their means decrease, so the result is
    the isotonic (non-decreasing) regression of the input under L2.
    """
    blocks = []
    for value in values:
        blocks.append([value, 1])
        while (len(blocks) > 1
               and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]):
            top = blocks.pop()
            blocks[-1][0] += top[0]
            blocks[-1][1] += top[1]
    return [total / count for total, count in blocks for _ in range(count)]


def _classify(name, platforms, note):
    """Derive (console-only?, unlock regime, prior-access?) from metadata text.

    Retrospective commentary (tiers, research notes, day-max, observed
    surges) is ignored; only release metadata up to the checkpoint is used.
    """
    text = note.split('Tier:')[0].split('Research note:')[0].lower()
    text = re.sub(r'no early[ -]access', '', text)
    early = ('early access' in text or 'early-access' in text
             or name == 'Saros')
    if platforms:
        console = not re.search(r'\bpc\b|stadia', platforms.lower())
    else:
        # Only the platform portion of the note may be consulted here.
        part = text.split('real unlock:')[0].split('source:')[0]
        console = ('console' in part or 'ps5' in part
                   or 'ps4' in part or 'switch' in part)
        if re.search(r'\bpc\b', part) and 'pc 2019' not in part:
            console = False
    rolling = 'midnight' in text and ('local' in text or 'rolling' in text)
    global_ = (('simultaneous' in text and 'no simultaneous' not in text)
               or 'single global' in text or 'pc single time' in text)
    if console and rolling:
        regime = 'rolling'
    elif rolling and global_:
        regime = 'mixed'
    elif rolling:
        regime = 'rolling' if console else 'mixed'
    elif global_:
        regime = 'global'
    elif console:
        regime = 'rolling'  # Explicit assumption; sensitivity-tested.
    else:
        regime = 'unknown'
    if name in ('Forza Horizon 5', 'Forza Horizon 6'):
        regime = 'mixed'
    if name in ('Diablo IV', 'Starfield', 'DOOM: The Dark Ages'):
        regime = 'global'
    return console, regime, bool(early)


def _build_curve(name, day, hourly, platforms='', note='', hours=None,
                 extra=False, regime_override=None):
    """One launch-day curve with its pre-checkpoint predictors.

    hours holds the 24 hourly viewer counts of `day` (hour 0 first). When
    hours is None they are read from the hourly history. Predictors use only
    data at or before the day's first hour: the log hour-1 count (x), the
    trend log(h1 / viewers one hour earlier), the pre-launch peak ratio and
    a channel-density term. The log ratios y[h] = log(viewers[h+1]) - x are
    attached only when all 24 hours are known (training curves).
    """
    day = day[:10]
    start = datetime.fromisoformat(day)
    by_hour = {p['from'][:13]: p for p in hourly if 'from' in p}
    if hours is None:
        hours = [by_hour.get('%sT%02d' % (day, h), {}).get('viewers')
                 for h in range(24)]
    first = hours[0]
    if first is None or not math.isfinite(float(first)) or first < 0:
        return None
    console, regime, early = _classify(name, platforms, note)
    if regime_override:
        regime = regime_override
        early = False
    cutoff = '%sT00' % day
    history = [p for p in hourly if p.get('from', '')[:13] <= cutoff]
    recent = [p for p in history
              if p['from'][:13] >= (start - timedelta(hours=12)).isoformat()[:13]]
    prev = by_hour.get((start - timedelta(hours=1)).isoformat()[:13], {})
    channels = by_hour.get(cutoff, {}).get('channels', 0) or 0
    lx = _log(first)
    # Predictors never look past the checkpoint.
    trend = _clip(lx - _log(prev.get('viewers', first)), -2, 2)
    prepeak = _clip(_log(max([p['viewers'] for p in recent] or [first])) - lx,
                    -2, 5)
    intensity = _clip(lx - _log(channels) - 4, -3, 3) if channels else 0.0
    curve = dict(name=name, group=name, day=day, hours=hours,
                 console=console, regime=regime, early=early,
                 is_extra=extra, weekday=start.weekday(), x=lx,
                 trend=trend, prepeak=prepeak, intensity=intensity,
                 platforms=platforms, unlock=note)
    if all(v is not None and math.isfinite(float(v)) and v >= 0
           for v in hours):
        curve['y'] = [_log(v) - lx for v in hours[1:]]
    return curve


def _training_curves(case, augment=True):
    """Historical launch curves from a case's comparables.

    Comparables that are not launches (target-history notes) are skipped.
    Documented first-access dates are added as extra curves. Every curve
    of the target game itself is removed, including other release dates.
    """
    curves = []
    for comp in case.get('comparables', []):
        unlock = comp.get('unlock', '')
        if ('target-history' in unlock.lower()
                or 'not a launch' in unlock.lower()):
            continue
        row = _build_curve(comp['name'], comp['day_start'], comp['hourly'],
                           comp.get('platforms', ''), unlock)
        if row and 'y' in row:
            curves.append(row)
        if augment and comp['name'] in EXTRA_DATES:
            date, regime = EXTRA_DATES[comp['name']]
            extra = _build_curve(comp['name'], date, comp['hourly'],
                                 comp.get('platforms', ''), unlock,
                                 extra=True, regime_override=regime)
            if extra and 'y' in extra:
                curves.append(extra)
    target_name = case.get('game', {}).get('name', '')
    return [r for r in curves if r['group'] != target_name]


def _target_curve(case):
    """The target launch as a curve: only hour 1 is known.

    The pre-checkpoint hourly history is used, with the hour-1 print placed
    at the day's hour 0 so the trend predictor sees the hour before it.
    """
    first = next(r for r in case['known'] if r['hour'] == 1)
    day = case['day_start'][:10]
    cutoff = '%sT01:00:00' % day
    hist = [p for p in case.get('history_hourly', [])
            if p.get('from', '') < cutoff]
    hist = [p for p in hist if p.get('from', '')[:13] != '%sT00' % day]
    hist.append(dict(first, **{'from': '%sT00:00:00Z' % day}))
    return _build_curve(case['game']['name'], day, hist,
                        note=case.get('notes', ''),
                        hours=[first['viewers']] + [None] * 23)


def _group_weights(rows):
    """Each game contributes total weight 1, however many dates it has."""
    counts = {}
    for row in rows:
        counts[row['group']] = counts.get(row['group'], 0) + 1
    return [1.0 / counts[row['group']] for row in rows]


def _baseline(target):
    """House baseline: every hour stays at hour 1, spread widens with time."""
    return [[target['x'] + z * 0.15 * math.sqrt(h) for z in ZVALUES]
            for h in range(1, 24)]


def _legacy_forecast(train, target):
    """Simple per-hour log-ratio regression on the clean legacy subset.

    Used only as a fallback when the trend regression has too few curves.
    """
    sub = [r for r in train
           if not r['is_extra']
           and r['hours'][0] / max(r['hours']) >= 0.08
           and 'early access' not in r['unlock'].lower()]
    if len(sub) < 3:
        return _baseline(target)
    xs = [r['x'] for r in sub]
    xm = st.mean(xs)
    sxx = sum((x - xm) ** 2 for x in xs)
    out = []
    for h in range(23):
        ys = [r['y'][h] for r in sub]
        ym = st.mean(ys)
        slope = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
                 / max(sxx, 1e-9))
        sd = math.sqrt(st.mean((y - ym - slope * (x - xm)) ** 2
                               for x, y in zip(xs, ys)))
        out.append([target['x'] + ym + slope * (target['x'] - xm)
                    + z * 0.7 * sd for z in ZVALUES])
    return out


def _weighted_template(train, target):
    """Similarity-weighted template: quantiles of past log-ratio curves.

    Curves are weighted by closeness in log hour-1, and penalised for
    differing unlock regime, console status and prior-access status.
    """
    if not train:
        return _baseline(target)
    weights = _group_weights(train)
    dists = []
    for row in train:
        d = ((row['x'] - target['x']) / 1.3) ** 2
        d += 1.5 * (row['regime'] != target['regime'])
        d += 0.8 * (row['console'] != target['console'])
        d += 1.2 * (row['early'] != target['early'])
        dists.append(d)
    nearest = min(dists)
    weights = [w * math.exp(-0.5 * (d - nearest))
               for w, d in zip(weights, dists)]
    return [[target['x'] + _weighted_quantile([r['y'][h] for r in train],
                                              weights, q)
             for q in LEVELS]
            for h in range(23)]


def _feature_values(row, keys, center, bounds):
    """Design-row values for the trend regression, intercept first."""
    x = _clip(row['x'], bounds[0] - 0.5, bounds[1] + 0.5) - center
    vals = {
        'x': x,
        'density': row['intensity'],
        'trend': row['trend'],
        'peak': row['prepeak'],
        'console': float(row['console']),
        'early': float(row['early']),
        'global': float(row['regime'] == 'global'),
        'trend_x': row['trend'] * (row['x'] - center),
        'early_x': float(row['early']) * (row['x'] - center),
    }
    return [1.0] + [vals[k] for k in keys]


def _trend_ridge(train, target, keys=('x', 'trend'), penalty=0.5,
                 spread=0.7, subset='clean25'):
    """Ridge regression of log ratios on log hour-1 and pre-checkpoint trend.

    One fit per forecast hour, on the clean training subset (no prior
    access, hour-1 at least 8% of the day's peak). Each game weighs the
    same in total. The intercept is effectively unpenalised (1e-7 jitter);
    other coefficients carry `penalty`. The residual scale is floored at
    0.10 and widened in quadrature when the target's hour-1 lies outside
    the training range.
    """
    if subset == 'clean25':
        sub = [r for r in train
               if not r['early'] and r['hours'][0] / max(r['hours']) >= 0.08]
    elif subset == 'live':
        sub = [r for r in train if r['hours'][0] / max(r['hours']) >= 0.08]
    else:
        sub = list(train)
    if len(sub) < 4:
        return _legacy_forecast(train, target)
    xs = [r['x'] for r in sub]
    center = st.mean(xs)
    bounds = (min(xs), max(xs))
    xx = [_feature_values(r, keys, center, bounds) for r in sub]
    xt = _feature_values(target, keys, center, bounds)
    gw = _group_weights(sub)
    p = len(xt)
    out = []
    for h in range(23):
        ys = [r['y'][h] for r in sub]
        mat = [[sum(w * v[i] * v[j] for v, w in zip(xx, gw))
                + (1e-7 if i == j == 0 else penalty if i == j else 0)
                for j in range(p)] for i in range(p)]
        rhs = [sum(w * v[j] * y for v, w, y in zip(xx, gw, ys))
               for j in range(p)]
        beta = _solve(mat, rhs)
        errors = [y - sum(a * b for a, b in zip(v, beta))
                  for v, y in zip(xx, ys)]
        mu = sum(a * b for a, b in zip(xt, beta))
        sigma = max(0.10, math.sqrt(
            sum(w * e * e for e, w in zip(errors, gw)) / sum(gw)))
        excess = max(0.0, target['x'] - bounds[1], bounds[0] - target['x'])
        sigma = math.hypot(sigma, 0.2 * excess)
        out.append([target['x'] + mu + spread * sigma * z for z in ZVALUES])
    return out


def _is_priority(row):
    """Console-only rolling-midnight first-access launch."""
    return row['console'] and row['regime'] == 'rolling' and not row['early']


def _balanced_weights(rows):
    """Half the total weight to the console cohort, half to large launches.

    Within each cohort every game weighs the same in total, however many
    dates it contributes. Rows in neither cohort get zero weight.
    """
    console_rows = [r for r in rows if _is_priority(r)]
    large_rows = [r for r in rows if r['hours'][0] > 250000]
    weights = [0.0] * len(rows)
    for cohort in (console_rows, large_rows):
        if not cohort:
            continue
        groups = {r['group'] for r in cohort}
        for i, row in enumerate(rows):
            if any(item is row for item in cohort):
                n = sum(x['group'] == row['group'] for x in cohort)
                weights[i] += 0.5 / (len(groups) * n)
    if not any(weights):
        return _group_weights(rows)
    return weights


def _trend_model(train, target, with_trend=True, penalty=0.5):
    """The V2 base forecast: trend ridge, or plain log-size ridge."""
    if with_trend:
        return _trend_ridge(train, target, keys=('x', 'trend'),
                            penalty=penalty, spread=0.7, subset='clean25')
    return _trend_ridge(train, target, keys=('x',),
                        penalty=1.0, spread=0.7, subset='clean25')


def _blend(first, second, weight):
    """Element-wise convex combination of two log-quantile forecasts."""
    return [[weight * a + (1.0 - weight) * b for a, b in zip(ra, rb)]
            for ra, rb in zip(first, second)]


def _calibrated_v2(train, target, penalty=0.5):
    """Balanced horizon calibration of the trend ridge forecast.

    Spread factors are chosen per horizon block (hours 2-6, 7-13, 14-24)
    from [0.65, 0.8, 1, 1.2, 1.5] by minimising the balanced inner
    leave-one-game-out pinball loss (console and large cohorts weighted
    equally, per game within each), plus a 0.01*log(k)^2 pull toward 1.
    The chosen factors rescale each hour's width around its median, and
    the resulting widths are pooled to be non-decreasing over the horizon.
    """
    relevant = [r for r in train
                if _is_priority(r) or r['hours'][0] > 250000]
    if len(relevant) < 4:
        return _trend_model(train, target, True, penalty)
    ws = _balanced_weights(relevant)
    pairs = []
    for held, w in zip(relevant, ws):
        rest = [r for r in train if r['group'] != held['group']]
        pairs.append((held, w, _trend_model(rest, held, True, penalty)))
    factors = []
    for lo, hi in ((0, 5), (5, 12), (12, 23)):
        scored = []
        for k in (0.65, 0.8, 1.0, 1.2, 1.5):
            total = 0.0
            for held, w, pp in pairs:
                for h in range(lo, hi):
                    y = _log(held['hours'][h + 1])
                    row = pp[h]
                    total += w * sum(
                        2 * max(t * (y - row[2] - k * (v - row[2])),
                                (t - 1) * (y - row[2] - k * (v - row[2])))
                        for t, v in zip(LEVELS, row))
            value = total / ((hi - lo) * 5 * sum(ws)) + 0.01 * math.log(k) ** 2
            scored.append((value, k))
        factors.append(min(scored)[1])
    pp = _trend_model(train, target, True, penalty)
    per_hour = [factors[0]] * 5 + [factors[1]] * 7 + [factors[2]] * 11
    widths = _pava([max(0.10, (row[4] - row[0]) / 3.29) * k
                    for row, k in zip(pp, per_hour)])
    return [[row[2] + z * s for z in ZVALUES] for row, s in zip(pp, widths)]


def _forecast_v2(train, target):
    """V2 routing: priority targets get the calibrated trend ridge.

    Membership uses only checkpoint information. Anything else falls back
    to the similarity-weighted template.
    """
    if _is_priority(target) or target['hours'][0] > 250000:
        return _calibrated_v2(train, target, penalty=0.5)
    return _weighted_template(train, target)


def _forecast_case(case):
    """Forecast one case file; returns the graded output dict."""
    if list(case['levels']) != list(LEVELS):
        raise ValueError('Unexpected quantile levels')
    train = _training_curves(case, augment=True)
    target = _target_curve(case)
    if len(train) < 4:
        pp = _baseline(target)
    else:
        pp = _forecast_v2(train, target)
    out = []
    for h, row in enumerate(pp, 2):
        # Stable positive values and monotone quantiles inside grader limits.
        qs = [math.floor(_clip(math.exp(_clip(v, -4.6, math.log(1e9))),
                               0.01, 1e9) * 100 + 0.5) / 100
              for v in sorted(row)]
        out.append({'hour': h, 'q': qs})
    return {'hours': out}


def main():
    src = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
    dst = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob('*.in')):
        case = json.loads(path.read_text(encoding='utf-8'))
        result = _forecast_case(case)
        (dst / (path.stem + '.out')).write_text(
            json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n',
            encoding='utf-8')


if __name__ == '__main__':
    main()
