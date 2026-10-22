"""Parameter optimisation: surrogate model + differential evolution (GA-like)."""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from common import add_features


def _frame(x, material, tool_dia, flutes, tool_wear):
    n = x.shape[1]
    return add_features(pd.DataFrame({
        "speed": x[0], "feed_per_tooth": x[1], "doc": x[2], "woc": x[3],
        "tool_wear": tool_wear, "tool_dia": tool_dia, "flutes": flutes,
        "material": material}, index=range(n)))


def optimize_parameters(model, material, tool_dia, flutes, tool_wear, bounds,
                        ra_limit=1.6, weight=0.5, seed=1):
    """weight=1 -> minimise Ra only; weight=0 -> maximise MRR (subject to Ra<=limit)."""
    mid = np.array([[(lo + hi) / 2] for lo, hi in bounds])
    ref = _frame(mid, material, tool_dia, flutes, tool_wear)
    ra_ref, mrr_ref = float(model.predict(ref)[0]), float(ref["mrr"][0])

    def obj(x):
        x = np.atleast_2d(x) if x.ndim == 1 else x
        if x.shape[0] != 4:
            x = x.T
        f = _frame(x, material, tool_dia, flutes, tool_wear)
        ra = model.predict(f)
        score = weight * ra / ra_ref - (1 - weight) * f["mrr"].values / mrr_ref
        return score + 100 * np.clip(ra - ra_limit, 0, None)

    r = differential_evolution(obj, bounds, seed=seed, maxiter=40, popsize=12,
                               vectorized=True, updating="deferred", tol=1e-4)
    best = _frame(np.array(r.x).reshape(4, 1), material, tool_dia, flutes, tool_wear)
    best["Ra_pred"] = model.predict(best)
    best["feasible"] = best["Ra_pred"] <= ra_limit
    return best.iloc[0]
