"""The metrics this study reports, and what each one hides.

Every function here takes an estimate and a reference and returns one number. They are gathered
in one file because several of them have properties that can lead to a wrong conclusion, and those
properties are documented beside the code.

The three that matter most are related by an identity. For series centred on their own means,

    anomaly R^2 = 2*r*alpha - alpha^2

where ``r`` is the correlation and ``alpha`` the ratio of standard deviations. So a single
coefficient near zero can arise from poor timing, from a wrong magnitude, or from both, and
reporting it alone conceals which. :func:`decompose` returns all three together for that reason.

A consequence worth knowing before adding a metric: a root-mean-square error normalised by the
reference's own standard deviation is ``sqrt(1 - anomaly_r2)``. It is not independent
information, and a table carrying both is a table with a redundant column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Skill", "anomaly_r2", "correlation", "amplitude_ratio", "decompose",
    "nse", "rmse", "rmse_bias_corrected", "kge2_change", "change_skill",
]

_MIN_N = 24          # below this, any of these is dominated by sampling noise


@dataclass(frozen=True)
class Skill:
    """Timing, magnitude and their combination, reported together."""
    n: int
    correlation: float
    amplitude: float
    anomaly_r2: float

    def __str__(self) -> str:
        return (f"n={self.n}  r={self.correlation:+.3f}  "
                f"alpha={self.amplitude:.3f}  R2a={self.anomaly_r2:+.3f}")


def _clean(est, ref):
    e = np.asarray(est, dtype=float)
    r = np.asarray(ref, dtype=float)
    if e.shape != r.shape:
        raise ValueError(f"estimate and reference differ in shape: {e.shape} vs {r.shape}")
    ok = np.isfinite(e) & np.isfinite(r)
    return e[ok], r[ok]


def anomaly_r2(est, ref, *, min_n: int = _MIN_N) -> float:
    """Coefficient of determination after centring both series on their own means.

    This is the study's headline measure. Two properties to keep in mind.

    It divides by the reference's own variance, so it is a ratio, not an error. A reservoir
    whose level barely moves has almost no variance to divide by, and any estimate of ordinary
    magnitude then produces a large negative value. Those are not failures of the estimate;
    they are the metric behaving as defined. Filter on the reference's variability before
    taking a minimum over a population.

    It is also not a rank: two estimates can have the same anomaly R^2 for opposite reasons.
    Use :func:`decompose` when the question is *what* changed.
    """
    e, r = _clean(est, ref)
    if len(r) < min_n:
        return np.nan
    ec, rc = e - e.mean(), r - r.mean()
    denom = np.sum(rc ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((ec - rc) ** 2) / denom)


def correlation(est, ref, *, min_n: int = _MIN_N) -> float:
    """Pearson correlation: whether the timing of the variation is reproduced.

    Insensitive to magnitude. An estimate at half the true amplitude but perfect timing
    correlates at 1.0.
    """
    e, r = _clean(est, ref)
    if len(r) < min_n or np.std(e) == 0 or np.std(r) == 0:
        return np.nan
    return float(np.corrcoef(e, r)[0, 1])


def amplitude_ratio(est, ref, *, min_n: int = _MIN_N) -> float:
    """Standard deviation of the estimate over that of the reference.

    One means the estimate varies as much as the reservoir actually does; below one it
    understates the variation. Says nothing about timing: a series with the right amplitude and
    random phase scores 1.0.
    """
    e, r = _clean(est, ref)
    if len(r) < min_n or np.std(r) == 0:
        return np.nan
    return float(np.std(e) / np.std(r))


def decompose(est, ref, *, min_n: int = _MIN_N) -> Skill:
    """Timing, magnitude and combined skill in one call.

    Prefer this to any single metric when reporting a comparison. The identity
    ``R2a = 2*r*alpha - alpha^2`` holds on the returned values to floating-point precision,
    so the first two explain the third rather than repeating it.
    """
    e, r = _clean(est, ref)
    return Skill(n=len(r),
                 correlation=correlation(e, r, min_n=min_n),
                 amplitude=amplitude_ratio(e, r, min_n=min_n),
                 anomaly_r2=anomaly_r2(e, r, min_n=min_n))


def nse(est, ref, *, min_n: int = _MIN_N) -> float:
    """Nash-Sutcliffe efficiency, on the series as given (not centred)."""
    e, r = _clean(est, ref)
    if len(r) < min_n:
        return np.nan
    denom = np.sum((r - r.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((e - r) ** 2) / denom)


def rmse(est, ref, *, min_n: int = _MIN_N) -> float:
    e, r = _clean(est, ref)
    if len(r) < min_n:
        return np.nan
    return float(np.sqrt(np.mean((e - r) ** 2)))


def rmse_bias_corrected(est, ref, *, min_n: int = _MIN_N) -> float:
    """RMSE after removing the constant offset between estimate and reference.

    The measure used in the global uncertainty accounting, because a constant per-reservoir
    offset cancels when storage is expressed as an anomaly and so should not enter the error
    budget.
    """
    e, r = _clean(est, ref)
    if len(r) < min_n:
        return np.nan
    d = e - r
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def kge2_change(r: float, alpha: float) -> float:
    """Two-component Kling-Gupta efficiency on a change series.

        1 - sqrt[(r - 1)^2 + (alpha - 1)^2]

    NOT the standard Kling-Gupta efficiency, and not taken from the literature. Standard KGE
    has three terms; the mean-ratio term is dropped here because the mean of a change series is
    close to zero, which makes that ratio unstable and meaningless. Define it in any write-up
    rather than citing it.

    Scale: one is perfect. An estimate that never changes has alpha = 0 and, by convention here,
    r = 0, giving -0.41 -- not zero. There is no natural zero point.
    """
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2))


def change_skill(est, ref, horizon_days: int, *, min_n: int = 60) -> Skill:
    """Skill of the estimated CHANGE over a given number of days.

    Differences the two series over ``horizon_days`` and scores the differences. This is the
    measure that separates an estimate which tracks the level from one that tracks movement: a
    series holding its last value between overpasses reproduces a slow level well while
    carrying no information about change at all.

    The returned ``anomaly_r2`` field holds :func:`kge2_change` of the differenced series, since
    that is the combination reported for change in this study.
    """
    e, r = _clean(est, ref)
    if len(r) <= horizon_days:
        return Skill(0, np.nan, np.nan, np.nan)
    de = e[horizon_days:] - e[:-horizon_days]
    dr = r[horizon_days:] - r[:-horizon_days]
    ok = np.isfinite(de) & np.isfinite(dr)
    if ok.sum() < min_n or np.std(dr[ok]) < 1e-9:
        return Skill(int(ok.sum()), np.nan, np.nan, np.nan)
    sd = np.std(de[ok])
    rr = float(np.corrcoef(de[ok], dr[ok])[0, 1]) if sd > 1e-12 else 0.0
    al = float(sd / np.std(dr[ok]))
    return Skill(int(ok.sum()), rr, al, kge2_change(rr, al))
