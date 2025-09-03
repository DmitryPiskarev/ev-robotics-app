import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import minimize

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

# ---------------- plotting helpers ----------------
def suspension_plotly(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist,
                      inner_x, inner_y, t_pickup, offset_dist):
    pos = suspension_positions(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
    if pos is None: return go.Figure()
    LCA_in, UCA_in, LCA_out, UCA_out = pos
    vec = UCA_out - LCA_out
    vec_perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
    outer = LCA_out + t_pickup * vec + offset_dist * vec_perp
    inner = np.array([inner_x, inner_y])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[LCA_in[0], LCA_out[0]], y=[LCA_in[1], LCA_out[1]],
                             mode="lines+markers", name="LCA"))
    fig.add_trace(go.Scatter(x=[UCA_in[0], UCA_out[0]], y=[UCA_in[1], UCA_out[1]],
                             mode="lines+markers", name="UCA"))
    fig.add_trace(go.Scatter(x=[LCA_out[0], UCA_out[0]], y=[LCA_out[1], UCA_out[1]],
                             mode="lines+markers", name="Knuckle"))
    fig.add_trace(go.Scatter(x=[inner[0], outer[0]], y=[inner[1], outer[1]],
                             mode="lines+markers", name="Tie-Rod"))
    fig.update_layout(title=f"Suspension Geometry at φ={phi_deg:.1f}°",
                      xaxis=dict(range=[-100, 200], scaleanchor="y", showgrid=False, zeroline=False),
                      yaxis=dict(range=[-50, 150], showgrid=False, zeroline=False),
                      margin=dict(l=10,r=10,t=40,b=10), height=400)
    return fig

def wheel_travel(phi_deg_range, l_lca, l_uca, inner_dist, ang_deg, outer_dist):
    travels = []
    for phi in phi_deg_range:
        pos = suspension_positions(phi, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
        if pos is None: continue
        _, _, _, UCA_out = pos
        travels.append(UCA_out[1])
    return phi_deg_range[:len(travels)], travels

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

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Bump Steer Optimizer", layout="wide")
st.title("Bump Steer Calculator & Optimizer")

with st.sidebar:
    st.header("Geometry Inputs")
    l_lca = st.number_input("Lower Control Arm length [mm]", 50.0, 200.0, 87.0, 1.0)
    l_uca = st.number_input("Upper Control Arm length [mm]", 50.0, 200.0, 74.0, 1.0)
    outer_dist = st.number_input("Knuckle distance (BJ-BJ) [mm]", 40.0, 120.0, 72.0, 1.0)
    inner_dist = st.number_input("Chassis pivot separation [mm]", 30.0, 100.0, 67.0, 1.0)

    st.header("Tie-Rod Inputs")
    inner_x = st.slider("Inner pivot X [mm]", -50.0, 50.0, 4.6, 0.1)
    inner_y = st.slider("Inner pivot Y [mm]", 0.0, 80.0, 19.3, 0.1)
    t_pickup = st.slider("Pickup (0=LCA, 1=UCA)", 0.0, 1.0, 0.36, 0.01)
    offset_dist = st.slider("Offset distance [mm]", -20.0, 20.0, -2.6, 0.1)
    ang_deg = st.slider("Chassis pivot inclination [deg]", -20.0, 20.0, -9.8, 0.1)

    st.header("Fix / Optimize Options")
    fix_inner = st.checkbox("Fix inner pivot X?", value=True)
    fix_inner_y = st.checkbox("Fix inner pivot Y?", value=False)
    fix_tie = st.checkbox("Fix tie-rod pickup?", value=False)
    fix_ang = st.checkbox("Fix chassis pivot inclination?", value=False)
    fix_offset = st.checkbox("Fix offset distance?", value=False)

    st.header("Suspension Motion")
    phi_deg = st.slider("Current LCA angle φ [deg]", -20.0, 20.0, 0.0, 0.5)

    if st.button("Auto-generate optimal configuration"):
        inner_fixed_val = inner_x if fix_inner else None
        inner_y_fixed_val = inner_y if fix_inner_y else None
        tie_fixed_val = t_pickup if fix_tie else None
        ang_fixed_val = ang_deg if fix_ang else None
        offset_fixed_val = offset_dist if fix_offset else None

        opt = optimize_suspension(l_lca, l_uca, outer_dist, inner_dist,
                                  inner_fixed_val, inner_y_fixed_val, tie_fixed_val,
                                  ang_fixed_val, offset_fixed_val)
        opt_inner_x, opt_inner_y, opt_t_pickup, opt_ang_deg, opt_offset = opt
        st.success(f"Optimized configuration:\n"
                   f"Inner pivot: ({opt_inner_x:.2f}, {opt_inner_y:.2f})\n"
                   f"Tie-Rod Pickup: {opt_t_pickup:.2f}\n"
                   f"Chassis Inclination: {opt_ang_deg:.2f}\n"
                   f"Offset: {opt_offset:.2f}")

    else:
        # If not optimizing, use current slider values
        opt_inner_x, opt_inner_y = inner_x, inner_y
        opt_t_pickup = t_pickup
        opt_ang_deg = ang_deg
        opt_offset = offset_dist

# ---------------- Layout with columns ----------------
col1, col2 = st.columns(2)

# Apply optimized values button
if st.button("Apply optimized values to sliders"):
    st.session_state['inner_x'] = opt_inner_x
    st.session_state['inner_y'] = opt_inner_y
    st.session_state['t_pickup'] = opt_t_pickup
    st.session_state['ang_deg'] = opt_ang_deg
    st.session_state['offset_dist'] = opt_offset

# Use session_state if available, otherwise optimized/current values
inner_val = st.session_state.get('inner_x', opt_inner_x)
inner_y_val = st.session_state.get('inner_y', opt_inner_y)
tie_pickup_val = st.session_state.get('t_pickup', opt_t_pickup)
ang_val = st.session_state.get('ang_deg', opt_ang_deg)
offset_val = st.session_state.get('offset_dist', opt_offset)

with col1:
    st.plotly_chart(
        suspension_plotly(phi_deg, l_lca, l_uca, inner_dist, ang_val, outer_dist,
                          inner_val, inner_y_val, tie_pickup_val, offset_val),
        use_container_width=True
    )

with col2:
    phi_vals, deviations = tie_rod_deviation(
        inner_xy=(inner_val, inner_y_val),
        t_on_knuckle=tie_pickup_val,
        l_lca=l_lca,
        l_uca=l_uca,
        inner_dist=inner_dist,
        ang_deg=ang_val,
        outer_dist=outer_dist,
        offset_dist=offset_val
    )

    max_idx = np.argmax(np.abs(deviations))
    min_idx = np.argmin(deviations)

    fig_dev = go.Figure()
    fig_dev.add_trace(go.Scatter(x=phi_vals, y=deviations, mode="lines", name="Δ Tie-Rod"))
    fig_dev.add_trace(go.Scatter(x=[phi_vals[max_idx]], y=[deviations[max_idx]],
                                 mode="markers+text", text=["Max"], textposition="top right",
                                 marker=dict(color="red", size=10)))
    fig_dev.add_trace(go.Scatter(x=[phi_vals[min_idx]], y=[deviations[min_idx]],
                                 mode="markers+text", text=["Min"], textposition="bottom right",
                                 marker=dict(color="blue", size=10)))
    fig_dev.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig_dev.update_layout(title="Tie-Rod Length Deviation vs LCA Angle",
                          xaxis_title="LCA angle φ [deg]",
                          yaxis_title="Δ Length [mm]",
                          height=400)
    st.plotly_chart(fig_dev, use_container_width=True)

    st.write(f"**Max Deviation:** {deviations[max_idx]:.2f} mm at φ={phi_vals[max_idx]:.1f}°")
    st.write(f"**Min Deviation:** {deviations[min_idx]:.2f} mm at φ={phi_vals[min_idx]:.1f}°")
    st.write(f"**Total Δ Tie-Rod Variation:** {deviations[max_idx]-deviations[min_idx]:.2f} mm")

    pos = suspension_positions(phi_deg, l_lca, l_uca, inner_dist, ang_val, outer_dist)
    if pos is not None:
        LCA_in, UCA_in, LCA_out, UCA_out = pos
        vec = UCA_out - LCA_out
        vec_perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
        outer = LCA_out + tie_pickup_val * vec + offset_val * vec_perp
        inner = np.array([inner_val, inner_y_val])
        tie_rod_vec = outer - inner
        tie_rod_angle = np.degrees(np.arctan2(tie_rod_vec[1], tie_rod_vec[0]))
        st.table({
            "Parameter": ["LCA Length", "UCA Length", "Tie-Rod Length", "Tie-Rod Angle"],
            "Value": [np.linalg.norm(LCA_out-LCA_in),
                      np.linalg.norm(UCA_out-UCA_in),
                      np.linalg.norm(tie_rod_vec),
                      f"{tie_rod_angle:.1f}°"]
        })

phi_range = np.linspace(-20, 20, 81)
phi_vals_travel, travels = wheel_travel(phi_range, l_lca, l_uca, inner_dist, ang_val, outer_dist)
fig_travel = go.Figure()
fig_travel.add_trace(go.Scatter(x=travels[:len(deviations)], y=deviations,
                                mode="lines", name="Δ Tie-Rod"))
fig_travel.update_layout(title="Tie-Rod Deviation vs Wheel Travel",
                         xaxis_title="Vertical Wheel Travel [mm]",
                         yaxis_title="Δ Tie-Rod Length [mm]",
                         height=400)
st.plotly_chart(fig_travel, use_container_width=True)

df = pd.DataFrame({"φ [deg]": phi_vals, "Δ Tie-Rod [mm]": deviations})
st.download_button("Download Tie-Rod Deviation CSV", df.to_csv(index=False), "tie_rod_deviation.csv")
