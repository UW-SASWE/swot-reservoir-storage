"""Causal outlier screens for the SWOT anchor's pass-level series.

Why these exist. A two-sided screen (`hampel`, `hampel_interp` in the trainer) judges each pass
against passes before AND after it, then repairs a rejected pass by interpolating toward the next
good one. The anchor at day x is the screened value of the latest pass p <= x, so it would depend on
passes after x. A screen for a nowcast may not do that.

Every screen here decides pass i from passes <= i only, so the daily anchor at day x depends on
nothing dated after x. Three rules are common to all of them, and are stated because they are
design choices and not properties of any one screen:

  1. Judged against ACCEPTED history only. A pass already rejected cannot vouch for a later one.
  2. Repair is carry-forward: a rejected pass takes the last accepted value. Interpolating toward a
     later pass would look ahead, which is the thing being removed.
  3. LOCK-IN GUARD. After MAX_CONSEC consecutive rejections the next pass is accepted regardless.
     Persistent disagreement with history is a real change in the reservoir, not an outlier, and
     without this a genuine step change would be rejected until the history caught up.

Each screen takes t (days, float), x (anchor value per pass) and a (raw per-pass area) and
returns (values, rejected): the repaired series and a boolean flag per pass.

The candidate set and the selection rule were fixed BEFORE any candidate was scored on the
training pool; see design_causal_screen.py.
"""
import numpy as np

MAX_CONSEC = 2
MIN_HIST = 3


def _mad_sigma(v):
    m = np.median(v)
    return m, 1.4826 * np.median(np.abs(v - m))


def _run(t, x, a, is_outlier):
    """Sequential driver. is_outlier(i, acc_idx) -> bool, where acc_idx holds the indices of
    passes accepted so far (all < i), so a screen cannot reach forward by construction."""
    n = len(x)
    acc_idx, rej = [], np.zeros(n, bool)
    vals = np.empty(n)
    consec = 0
    for i in range(n):
        bad = False
        if len(acc_idx) >= MIN_HIST and consec < MAX_CONSEC:
            bad = bool(is_outlier(i, acc_idx))
        if bad:
            rej[i] = True
            vals[i] = x[acc_idx[-1]]
            consec += 1
        else:
            acc_idx.append(i)
            vals[i] = x[i]
            consec = 0
    return vals, rej


def screen_none(t, x, a):
    return np.asarray(x, float).copy(), np.zeros(len(x), bool)


def screen_hampel1(t, x, a, k=3, nsig=3.0):
    """C1. One-sided Hampel: |x - median(last k accepted)| > nsig * 1.4826 * MAD."""
    x = np.asarray(x, float)

    def out(i, acc):
        w = x[acc[-k:]]
        if len(w) < MIN_HIST:
            return False
        med, s = _mad_sigma(w)
        return s > 0 and abs(x[i] - med) > nsig * s
    return _run(t, x, a, out)


def screen_increment(t, x, a, nsig=3.0, npairs=8):
    """C2. Increment gate, robust to a steady trend. The step from the last accepted pass, per
    square-root day of gap, is compared with the recent steps between accepted passes. A
    filling or drawdown trend makes past steps consistently signed, so it does not read as an
    outlier the way it does against a one-sided median."""
    x = np.asarray(x, float); t = np.asarray(t, float)

    def out(i, acc):
        if len(acc) < npairs // 2 + 1:
            return False
        idx = acc[-(npairs + 1):]
        dt = np.maximum(np.diff(t[idx]), 1.0)
        z = np.diff(x[idx]) / np.sqrt(dt)
        if len(z) < 4:
            return False
        med, s = _mad_sigma(z)
        j = acc[-1]
        zi = (x[i] - x[j]) / np.sqrt(max(t[i] - t[j], 1.0))
        return s > 0 and abs(zi - med) > nsig * s
    return _run(t, x, a, out)


def screen_coverage(t, x, a, tau=0.5, nhist=5):
    """C3. Partial-observation gate: reject a pass whose raw area is far below the recent
    accepted median. A pass that saw only part of the reservoir has a small area and an
    unrepresentative level. Uses same-pass information and accepted history only."""
    x = np.asarray(x, float); a = np.asarray(a, float)

    def out(i, acc):
        h = a[acc[-nhist:]]
        if len(h) < MIN_HIST:
            return False
        m = np.median(h)
        return m > 0 and a[i] / m < tau
    return _run(t, x, a, out)


def screen_union(t, x, a, screens):
    """C4. Reject if ANY listed screen would. Built by composing the decision rules."""
    x = np.asarray(x, float); t = np.asarray(t, float); a = np.asarray(a, float)
    n = len(x)
    acc_idx, rej = [], np.zeros(n, bool)
    vals = np.empty(n)
    consec = 0
    # each rule is re-evaluated on the shared accepted history
    rules = [_rule(name, kw, x, t, a) for name, kw in screens]
    for i in range(n):
        bad = False
        if len(acc_idx) >= MIN_HIST and consec < MAX_CONSEC:
            bad = any(r(i, acc_idx) for r in rules)
        if bad:
            rej[i] = True
            vals[i] = x[acc_idx[-1]]
            consec += 1
        else:
            acc_idx.append(i)
            vals[i] = x[i]
            consec = 0
    return vals, rej


def _rule(name, kw, x, t, a):
    """The bare decision function of a named screen, without its driver."""
    if name == 'increment':
        nsig, npairs = kw.get('nsig', 3.0), kw.get('npairs', 8)

        def r(i, acc):
            if len(acc) < npairs // 2 + 1:
                return False
            idx = acc[-(npairs + 1):]
            z = np.diff(x[idx]) / np.sqrt(np.maximum(np.diff(t[idx]), 1.0))
            if len(z) < 4:
                return False
            med, s = _mad_sigma(z)
            j = acc[-1]
            zi = (x[i] - x[j]) / np.sqrt(max(t[i] - t[j], 1.0))
            return s > 0 and abs(zi - med) > nsig * s
        return r
    if name == 'coverage':
        tau, nh = kw.get('tau', 0.5), kw.get('nhist', 5)

        def r(i, acc):
            h = a[acc[-nh:]]
            if len(h) < MIN_HIST:
                return False
            m = np.median(h)
            return m > 0 and a[i] / m < tau
        return r
    raise ValueError(name)


# ---- registry: the candidate set, fixed in advance ---------------------------------------------
CANDIDATES = {
    'C0_none':        lambda t, x, a: screen_none(t, x, a),
    'C1_hampel1':     lambda t, x, a: screen_hampel1(t, x, a, k=3, nsig=3.0),
    'C2a_incr_n3':    lambda t, x, a: screen_increment(t, x, a, nsig=3.0),
    'C2b_incr_n4':    lambda t, x, a: screen_increment(t, x, a, nsig=4.0),
    'C3a_cov_0.5':    lambda t, x, a: screen_coverage(t, x, a, tau=0.5),
    'C3b_cov_0.7':    lambda t, x, a: screen_coverage(t, x, a, tau=0.7),
    'C4_incr4+cov.5': lambda t, x, a: screen_union(
        t, x, a, [('increment', {'nsig': 4.0}), ('coverage', {'tau': 0.5})]),
}


def causal_screen(name, t, x, a):
    """Entry point used by build_daily: returns (repaired values, rejected flags)."""
    return CANDIDATES[name](t, x, a)
