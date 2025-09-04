import math
from .main import suspension_positions


# ---------------- geometry helpers ----------------
def circle_intersections(p0, r0, p1, r1):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d > r0 + r1 or d < abs(r0 - r1) or (d == 0 and r0 == r1):
        return []
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h2 = r0**2 - a**2
    if h2 < 0: h2 = 0
    h = math.sqrt(h2)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def get_coords_from_y(init, l, angle_deg):
    rad = math.radians(angle_deg)
    return init[0] - l * math.sin(rad), init[1] + l * math.cos(rad)


def get_coords_x_frame(init, l, angle_deg):
    rad = math.radians(angle_deg)
    return init[0] + l * math.cos(rad), init[1] - l * math.sin(rad)


def wheel_travel(phi_deg_range, l_lca, l_uca, inner_dist, ang_deg, outer_dist):
    travels = []
    for phi in phi_deg_range:
        pos = suspension_positions(phi, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
        if pos is None: continue
        _, _, _, UCA_out = pos
        travels.append(UCA_out[1])
    return phi_deg_range[:len(travels)], travels
