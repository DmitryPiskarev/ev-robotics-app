import numpy as np
from scipy.optimize import minimize


# ---------------- multi-parameter optimizer ----------------
def optimize_suspension(l_lca, l_uca, outer_dist, inner_dist,
                        inner_fixed=None, inner_y_fixed=None, tie_fixed=None,
                        ang_fixed=None, offset_fixed=None):
    def objective(vars):
        inner_x, inner_y, t_pickup, ang_deg, offset_dist = vars
        inner_x_val = inner_x if inner_fixed is None else inner_fixed
        inner_y_val = inner_y if inner_y_fixed is None else inner_y_fixed
        t_val = t_pickup if tie_fixed is None else tie_fixed
        ang_val = ang_deg if ang_fixed is None else ang_fixed
        offset_val = offset_dist if offset_fixed is None else offset_fixed
        _, deviations = tie_rod_deviation(
            inner_xy=(inner_x_val, inner_y_val),
            t_on_knuckle=t_val,
            l_lca=l_lca,
            l_uca=l_uca,
            inner_dist=inner_dist,
            ang_deg=ang_val,
            outer_dist=outer_dist,
            offset_dist=offset_val
        )
        return np.max(np.abs(deviations))

    x0 = [
        0.0 if inner_fixed is None else inner_fixed,
        19.3 if inner_y_fixed is None else inner_y_fixed,
        0.36 if tie_fixed is None else tie_fixed,
        -9.8 if ang_fixed is None else ang_fixed,
        -2.6 if offset_fixed is None else offset_fixed
    ]

    bounds = [
        (-50, 50) if inner_fixed is None else (inner_fixed, inner_fixed),
        (0, 80) if inner_y_fixed is None else (inner_y_fixed, inner_y_fixed),
        (0, 1) if tie_fixed is None else (tie_fixed, tie_fixed),
        (-20, 20) if ang_fixed is None else (ang_fixed, ang_fixed),
        (-20, 20) if offset_fixed is None else (offset_fixed, offset_fixed)
    ]

    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    return res.x
