from .geometry import get_coords_from_y, get_coords_x_frame, circle_intersections
import numpy as np
from scipy.optimize import minimize


def suspension_positions(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist):
    LCA_inner = np.array([0.0, 0.0])
    UCA_inner = np.array(get_coords_from_y((0, 0), inner_dist, ang_deg))
    LCA_outer = np.array(get_coords_x_frame((0, 0), l_lca, phi_deg))
    sols = circle_intersections(UCA_inner, l_uca, LCA_outer, outer_dist)
    if not sols: return None
    UCA_outer = np.array(max(sols, key=lambda p: p[1]))
    return LCA_inner, UCA_inner, LCA_outer, UCA_outer


def tie_rod_deviation(inner_xy, t_on_knuckle, l_lca, l_uca,
                      inner_dist, ang_deg, outer_dist, offset_dist=0,
                      phi_range=np.linspace(-20, 20, 81)):
    inner_xy = np.asarray(inner_xy, float)
    lengths, phis_ok = [], []
    for phi in phi_range:
        pos = suspension_positions(phi, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
        if pos is None: continue
        LCA_in, UCA_in, LCA_out, UCA_out = pos
        vec = UCA_out - LCA_out
        vec_perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
        outer = LCA_out + t_on_knuckle * vec + offset_dist * vec_perp
        L = np.linalg.norm(outer - inner_xy)
        lengths.append(L)
        phis_ok.append(phi)
    lengths = np.array(lengths)
    L0 = lengths[np.argmin(np.abs(np.array(phis_ok)))]
    return np.array(phis_ok), lengths - L0


def wheel_travel(phi_deg_range, l_lca, l_uca, inner_dist, ang_deg, outer_dist):
    travels = []
    for phi in phi_deg_range:
        pos = suspension_positions(phi, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
        if pos is None: continue
        _, _, _, UCA_out = pos
        travels.append(UCA_out[1])
    return phi_deg_range[:len(travels)], travels


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

