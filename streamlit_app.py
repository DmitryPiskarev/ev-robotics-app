import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from utils.main import tie_rod_deviation, suspension_positions, wheel_travel, optimize_suspension


# ---------------- plotting helpers ----------------
def suspension_plotly(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist,
                      inner_x, inner_y, t_pickup, offset_dist, height=600):
    pos = suspension_positions(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
    if pos is None: return go.Figure()
    LCA_in, UCA_in, LCA_out, UCA_out = pos
    vec = UCA_out - LCA_out
    vec_perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
    outer = LCA_out + t_pickup * vec + offset_dist * vec_perp
    inner = np.array([inner_x, inner_y])
    fig = go.Figure()

    fig.add_annotation(
        x=-20, y=-40,
        text="Chassis / Body side",
        showarrow=False,
        font=dict(size=12, color="gray"),
        xanchor="right"
    )

    fig.add_annotation(
        x=180, y=-40,
        text="Wheel / Outer side",
        showarrow=False,
        font=dict(size=12, color="gray"),
        xanchor="left"
    )

    fig.add_hline(y=0, line=dict(color="black", dash="dot"), opacity=0.4)

    fig.add_trace(go.Scatter(
        x=[LCA_in[0], UCA_in[0]], y=[LCA_in[1], UCA_in[1]],
        mode="markers", name="Inner Pivots",
        marker=dict(symbol="circle", size=10, color="black")
    ))

    fig.add_trace(go.Scatter(
        x=[LCA_out[0], UCA_out[0]], y=[LCA_out[1], UCA_out[1]],
        mode="markers", name="Outer Pivots",
        marker=dict(symbol="circle-open", size=10, color="red")
    ))

    fig.add_annotation(
        x=200, y=50,
        ax=250, ay=50,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowcolor="blue"
    )
    fig.add_annotation(
        x=260, y=50,
        text="Wheel side",
        showarrow=False,
        font=dict(size=12, color="blue"),
        xanchor="left"
    )

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
                      margin=dict(l=10, r=10, t=40, b=10),
                      height=height)

    fig.update_layout(
        xaxis=dict(range=[-50, 250], scaleanchor="y"),
        yaxis=dict(range=[-50, 150])
    )

    return fig


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
    inner_x = st.number_input("Inner pivot X [mm]", -50.0, 50.0, 4.6, step=0.01, format="%.2f")
    inner_y = st.number_input("Inner pivot Y [mm]", 0.0, 80.0, 19.3, step=0.01, format="%.2f")
    t_pickup = st.number_input("Pickup (0=LCA, 1=UCA)", 0.0, 1.0, 0.36, step=0.01, format="%.2f")
    offset_dist = st.number_input("Offset distance [mm]", -20.0, 20.0, -2.6, step=0.01, format="%.2f")
    ang_deg = st.number_input("Chassis pivot inclination [deg]", -20.0, 20.0, -9.8, step=0.01, format="%.2f")

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


with col1:
    st.plotly_chart(
        suspension_plotly(phi_deg, l_lca, l_uca, inner_dist, opt_ang_deg, outer_dist,
                          opt_inner_x, opt_inner_y, opt_t_pickup, opt_offset, height=600),
        use_container_width=True
    )

with col2:
    phi_vals, deviations = tie_rod_deviation(
        inner_xy=(opt_inner_x, opt_inner_y),
        t_on_knuckle=opt_t_pickup,
        l_lca=l_lca,
        l_uca=l_uca,
        inner_dist=inner_dist,
        ang_deg=opt_ang_deg,
        outer_dist=outer_dist,
        offset_dist=opt_offset
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

    pos = suspension_positions(phi_deg, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
    if pos is not None:
        LCA_in, UCA_in, LCA_out, UCA_out = pos
        vec = UCA_out - LCA_out
        vec_perp = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)
        outer = LCA_out + t_pickup * vec + offset_dist * vec_perp
        inner = np.array([inner_x, inner_y])
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
phi_vals_travel, travels = wheel_travel(phi_range, l_lca, l_uca, inner_dist, ang_deg, outer_dist)
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
