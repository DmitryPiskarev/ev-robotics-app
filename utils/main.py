from utils.geometry import get_coords_from_y, get_coords_x_frame, circle_intersections
import numpy as np


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
