import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go


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
    if not sols:
        return None
    UCA_outer = np.array(max(sols, key=lambda p: p[1]))
    return LCA_inner, UCA_inner, LCA_outer, UCA_outer


def tie_rod_deviation(inner_xy, t_on_knuckle, l_lca, l_uca,
                      inner_dist, ang_deg, outer_dist, offset_dist=0,
                      phi_range=np.linspace(-20, 20, 81)):
    inner_xy = np.asarray(inner_xy, float)
    lengths, phis_ok = [], []
    for phi in phi_range:
        pos = suspension_positions(phi, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
        if pos is None:
            continue
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
    if pos is None:
        return go.Figure()

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

    fig.update_layout(
        title=f"Suspension Geometry at φ={phi_deg:.1f}°",
        xaxis=dict(range=[-100, 200], scaleanchor="y", showgrid=False, zeroline=False),
        yaxis=dict(range=[-50, 150], showgrid=False, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400
    )
    return fig


def deviation_plotly(inner_x, inner_y, t_pickup, l_lca, l_uca, inner_dist,
                     ang_deg, outer_dist, offset_dist):
    phi_vals, deviations = tie_rod_deviation(
        (inner_x, inner_y), t_pickup,
        l_lca, l_uca, inner_dist, ang_deg,
        outer_dist, offset_dist
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi_vals, y=deviations, mode="lines", name="Δ Length"))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(
        title="Tie-Rod Length Deviation vs LCA Angle",
        xaxis_title="LCA angle φ [deg]",
        yaxis_title="Δ Length [mm]",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400
    )
    return fig


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Bump Steer Visualizer", layout="wide")
st.title("Bump Steer Calculator & Visualizer")

with st.sidebar:
    st.header("Geometry Inputs")
    l_lca = st.number_input("Lower Control Arm length [mm]", 50.0, 200.0, 87.0, 1.0)
    l_uca = st.number_input("Upper Control Arm length [mm]", 50.0, 200.0, 74.0, 1.0)
    outer_dist = st.number_input("Knuckle distance (BJ-BJ) [mm]", 40.0, 120.0, 72.0, 1.0)
    inner_dist = st.number_input("Chassis pivot separation [mm]", 30.0, 100.0, 67.0, 1.0)
    ang_deg = st.slider("Chassis pivot inclination [deg]", -20.0, 20.0, -9.8, 0.1)

    st.header("Tie-Rod Inputs")
    inner_x = st.slider("Inner pivot X [mm]", -50.0, 50.0, 4.6, 0.1)
    inner_y = st.slider("Inner pivot Y [mm]", 0.0, 80.0, 19.3, 0.1)
    t_pickup = st.slider("Pickup (0=LCA, 1=UCA)", 0.0, 1.0, 0.36, 0.01)
    offset_dist = st.slider("Offset distance [mm]", -20.0, 20.0, -2.6, 0.1)

    st.header("Suspension Motion")
    phi_deg = st.slider("Current LCA angle φ [deg]", -20.0, 20.0, 0.0, 0.5)


# ---------------- Layout with columns ----------------
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        suspension_plotly(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist,
                          inner_x, inner_y, t_pickup, offset_dist),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        deviation_plotly(inner_x, inner_y, t_pickup, l_lca, l_uca,
                         inner_dist, ang_deg, outer_dist, offset_dist),
        use_container_width=True
    )
