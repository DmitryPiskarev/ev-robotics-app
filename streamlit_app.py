import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

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
    if h2 < 0:
        h2 = 0
    h = math.sqrt(h2)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry =  dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]

def get_coords_from_y(init, l, angle_deg):
    rad = math.radians(angle_deg)
    return init[0] - l * math.sin(rad), init[1] + l * math.cos(rad)

def get_coords_x_frame(init, l, angle_deg):
    rad = math.radians(angle_deg)
    return init[0] - l * math.cos(rad), init[1] - l * math.sin(rad)

# ------------- suspension model -------------------
def suspension_positions(phi_deg, l_lca, l_uca, outer_dist, inner_dist, ang_deg):
    LCA_inner = np.array([0.0, 0.0])
    UCA_inner = np.array(get_coords_from_y((0, 0), inner_dist, ang_deg))

    LCA_outer = np.array(get_coords_x_frame((0, 0), l_lca, phi_deg))
    sols = circle_intersections(UCA_inner, l_uca, LCA_outer, outer_dist)
    if not sols:
        return None
    UCA_outer = np.array(max(sols, key=lambda p: p[1]))
    return LCA_outer, UCA_outer

def tie_rod_deviation(inner_xy, t_on_knuckle, l_lca, l_uca, outer_dist, inner_dist, ang_deg, offset_dist=0, phi_range=np.linspace(-20, 20, 81)):
    inner_xy = np.asarray(inner_xy, float)
    lengths = []
    phis_ok = []
    for phi in phi_range:
        pos = suspension_positions(phi, l_lca, l_uca, outer_dist, inner_dist, ang_deg)
        if pos is None:
            continue
        LCA_out, UCA_out = pos
        vec = UCA_out - LCA_out
        vec_perp = np.array([-vec[1], vec[0]])
        vec_perp = vec_perp / np.linalg.norm(vec_perp)
        outer = LCA_out + t_on_knuckle * vec + offset_dist * vec_perp
        L = np.linalg.norm(outer - inner_xy)
        lengths.append(L)
        phis_ok.append(phi)
    lengths = np.array(lengths)
    L0 = lengths[np.argmin(np.abs(np.array(phis_ok)))]
    return np.array(phis_ok), lengths - L0

# ---------------- streamlit app -------------------
st.set_page_config(page_title="Bump Steer Calculator", layout="wide")
st.title("🚗 Bump Steer Calculator & Visualizer")

st.sidebar.header("Suspension Parameters")
l_lca = st.sidebar.number_input("Lower Control Arm length [mm]", 50.0, 200.0, 87.0, step=0.1)
l_uca = st.sidebar.number_input("Upper Control Arm length [mm]", 50.0, 200.0, 77.0, step=0.1)
outer_dist = st.sidebar.number_input("Outer BJ distance [mm]", 50.0, 200.0, 72.0, step=0.1)
inner_dist = st.sidebar.number_input("Inner pivot distance [mm]", 20.0, 200.0, 58.0, step=0.1)
ang_deg = st.sidebar.slider("Inner pivot angle [deg]", -15.0, 15.0, 9.8, step=0.1)

st.sidebar.header("Tie-rod Parameters")
inner_x = st.sidebar.slider("Tie-rod inner X [mm]", -50.0, 50.0, -3.9, step=0.1)
inner_y = st.sidebar.slider("Tie-rod inner Y [mm]", 0.0, 100.0, 16.5, step=0.1)
t_pickup = st.sidebar.slider("Tie-rod pickup factor", 0.0, 1.0, 0.296, step=0.001)
offset_dist = st.sidebar.slider("Offset distance [mm]", -20.0, 20.0, 0.0, step=0.1)

phi_vals, deviations = tie_rod_deviation((inner_x, inner_y), t_pickup, l_lca, l_uca, outer_dist, inner_dist, ang_deg, offset_dist)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(phi_vals, deviations, lw=2)
ax.axhline(0, color='k', ls='--')
ax.set_title("Tie-rod length deviation vs LCA rotation")
ax.set_xlabel("LCA angle (deg)")
ax.set_ylabel("Δ Length (mm)")
ax.grid(True)

st.pyplot(fig)

st.markdown("---")
st.info("Move the sliders in the sidebar to see how suspension geometry affects bump steer.")
