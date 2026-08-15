# =============================================================================
# dashboard.py — Streamlit dashboard for Cloud Resource Allocation
# =============================================================================
# Run: streamlit run dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Cloud Resource Allocation",
    page_icon  = "☁️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    div[data-testid="metric-container"] {
        background-color: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 8px;
        padding: 16px;
    }
    h1 { color: #4f9cf9; font-family: 'Courier New', monospace; }
    h2 { color: #c8d3f5; }
    h3 { color: #a9b1d6; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_results():
    path = os.path.join(RESULTS_DIR, "comparison.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_q_hats():
    path = os.path.join(BASE_DIR, "checkpoints", "q_hats.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {"cpu": 0.0, "mem": 0.0}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☁️ Cloud Allocation")
    st.markdown("**LGBM + CQR Pipeline**")
    st.markdown("---")

    q_hats = load_q_hats()
    st.markdown("### Model Config")
    st.markdown(f"**q_hat CPU** : `{q_hats['cpu']:.6f}`")
    st.markdown(f"**q_hat MEM** : `{q_hats['mem']:.6f}`")
    st.markdown(f"**Horizon**   : `2 ticks (10 min)`")
    st.markdown(f"**Algorithm** : `LightGBM + CQR`")
    st.markdown("---")

    st.markdown("### Display")
    n_ticks     = st.slider("Ticks to display", 100, 5000, 1000, 100)
    show_raw    = st.checkbox("Show raw data", False)
    auto_refresh= st.checkbox("Auto refresh (30s)", False)

    st.markdown("---")
    st.markdown("### AWS Pricing (ap-south-1)")
    st.markdown("""
| Tier | Instance | $/hr |
|------|----------|------|
| XSMALL | t2.nano | $0.0058 |
| SMALL | t2.micro | $0.0116 |
| MEDIUM | t2.small | $0.0230 |
| LARGE | t2.medium | $0.0464 |
""")

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_results()

if df is None:
    st.error("No results found. Run `python pipeline.py --run` first.")
    st.stop()

df_display = df.iloc[:n_ticks].copy().reset_index(drop=True)
t          = df_display.index.values

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ☁️ Cloud Resource Allocation Dashboard")
st.markdown(
    f"**Google Cluster 2019 · LightGBM + Conformalized Quantile Regression"
    f" · {len(df):,} ticks simulated**"
)
st.markdown("---")

# ── KPI metrics ───────────────────────────────────────────────────────────────
sla_rate   = float(df["sla_violation"].mean())
coverage   = 1.0 - sla_rate
over_prov  = float(df["over_prov"].mean())
total_cost = float(df["aws_total_cost"].sum()
                   if "aws_total_cost" in df.columns
                   else df["cost"].sum())
vm_mean    = float(df["vm_count"].mean()
                   if "vm_count" in df.columns else 0)
scale_ups  = int(df["scaled_up"].sum()
                 if "scaled_up" in df.columns else 0)

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    st.metric("SLA Violation", f"{sla_rate*100:.2f}%",
              delta=f"{(sla_rate-0.10)*100:.2f}% vs 10% target",
              delta_color="inverse")
    # Add breakdown below metric
    if "sla_viol_cpu" in df.columns:
        cpu_viol = float(df["sla_viol_cpu"].mean())
        mem_viol = float(df["sla_viol_mem"].mean())
        both_viol = float(((df["sla_viol_cpu"]==1) &
                           (df["sla_viol_mem"]==1)).mean())
        st.caption(f"CPU: {cpu_viol*100:.2f}% | "
                   f"MEM: {mem_viol*100:.2f}% | "
                   f"Both: {both_viol*100:.2f}%")
with c2:
    st.metric("Coverage", f"{coverage*100:.2f}%",
              delta=f"+{(coverage-0.90)*100:.2f}% vs 90% target")
with c3:
    st.metric("Over-provisioning", f"{over_prov:.4f}")
with c4:
    st.metric("Total AWS Cost", f"${total_cost:.2f}")
with c5:
    st.metric("Avg Active VMs", f"{vm_mean:.1f}")
with c6:
    st.metric("Scale Events", f"{scale_ups}")

st.markdown("---")

# ── Chart 1: Actual vs Allocated CPU ──────────────────────────────────────────
st.markdown("## 📈 Actual vs Allocated CPU")

fig1 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.08,
    subplot_titles=("CPU Utilisation", "Rolling SLA Violation Rate (100-tick window)")
)

# Panel 1 — actual fill
fig1.add_trace(go.Scatter(
    x=t, y=df_display["actual_cpu"],
    name="Actual CPU",
    fill="tozeroy",
    fillcolor="rgba(79,156,249,0.25)",
    line=dict(color="#4f9cf9", width=1.2),
), row=1, col=1)

# Allocated line
fig1.add_trace(go.Scatter(
    x=t, y=df_display["alloc_cpu"],
    name="Allocated (CQR)",
    line=dict(color="#f7768e", width=2, dash="dash"),
), row=1, col=1)

# Point prediction
if "point_cpu" in df_display.columns:
    fig1.add_trace(go.Scatter(
        x=t, y=df_display["point_cpu"],
        name="Point Prediction",
        line=dict(color="#e0af68", width=1, dash="dot"),
        opacity=0.6,
    ), row=1, col=1)

# SLA violation markers
viol_idx = df_display[df_display["sla_violation"] == 1].index
if len(viol_idx):
    fig1.add_trace(go.Scatter(
        x=viol_idx,
        y=df_display.loc[viol_idx, "actual_cpu"],
        mode="markers",
        name="SLA Violation",
        marker=dict(color="#ff4444", size=5, symbol="x"),
    ), row=1, col=1)

# Panel 2 — smooth rolling SLA (100-tick window)
rolling_sla = (df_display["sla_violation"]
               .rolling(100, min_periods=1)
               .mean())
fig1.add_trace(go.Scatter(
    x=t, y=rolling_sla,
    name="Rolling SLA",
    fill="tozeroy",
    fillcolor="rgba(247,118,142,0.25)",
    line=dict(color="#f7768e", width=1.5),
), row=2, col=1)

fig1.add_hline(
    y=0.10, line_dash="dash", line_color="#ffffff",
    line_width=1,
    annotation_text="10% SLA target",
    annotation_font_color="#ffffff",
    row=2, col=1,
)

fig1.update_layout(
    height=520,
    paper_bgcolor="#0f1117",
    plot_bgcolor="#1a1d27",
    font=dict(color="#a9b1d6", size=12),
    legend=dict(
        bgcolor="#1a1d27", bordercolor="#2d3147",
        orientation="h", yanchor="bottom",
        y=1.02, xanchor="right", x=1
    ),
    margin=dict(l=50, r=20, t=40, b=40),
)
fig1.update_xaxes(gridcolor="#2d3147", zerolinecolor="#2d3147",
                  showline=True, linecolor="#2d3147")
fig1.update_yaxes(gridcolor="#2d3147", zerolinecolor="#2d3147")
fig1.update_yaxes(title_text="CPU Utilisation", row=1, col=1)
fig1.update_yaxes(title_text="SLA Rate", row=2, col=1)
fig1.update_xaxes(title_text="Tick (each = 5 min)", row=2, col=1)

st.plotly_chart(fig1, use_container_width=True)

# CPU vs MEM SLA breakdown chart
st.markdown("## 🔴 CPU vs MEM SLA Breakdown")

col_cpu, col_mem = st.columns(2)

with col_cpu:
    st.markdown("### CPU Allocation")
    fig_cpu = go.Figure()
    fig_cpu.add_trace(go.Scatter(
        x=t, y=df_display["actual_cpu"],
        name="Actual CPU",
        fill="tozeroy",
        fillcolor="rgba(79,156,249,0.25)",
        line=dict(color="#4f9cf9", width=1.2),
    ))
    fig_cpu.add_trace(go.Scatter(
        x=t, y=df_display["alloc_cpu"],
        name="Allocated CPU",
        line=dict(color="#f7768e", width=2, dash="dash"),
    ))
    # CPU violations
    cpu_viol_idx = df_display[df_display["sla_viol_cpu"]==1].index
    if len(cpu_viol_idx):
        fig_cpu.add_trace(go.Scatter(
            x=cpu_viol_idx,
            y=df_display.loc[cpu_viol_idx, "actual_cpu"],
            mode="markers",
            name="CPU Violation",
            marker=dict(color="#ff4444", size=5, symbol="x"),
        ))
    fig_cpu.update_layout(
        height=280,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#a9b1d6"),
        legend=dict(bgcolor="#1a1d27"),
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis_title="Tick",
        yaxis_title="CPU Utilisation",
    )
    fig_cpu.update_xaxes(gridcolor="#2d3147")
    fig_cpu.update_yaxes(gridcolor="#2d3147")
    st.plotly_chart(fig_cpu, use_container_width=True)
    st.caption(f"CPU SLA violation: {float(df['sla_viol_cpu'].mean())*100:.2f}%")

with col_mem:
    st.markdown("### MEM Allocation")
    if "alloc_mem" in df_display.columns:
        fig_mem = go.Figure()
        fig_mem.add_trace(go.Scatter(
            x=t, y=df_display["actual_mem"],
            name="Actual MEM",
            fill="tozeroy",
            fillcolor="rgba(158,206,106,0.25)",
            line=dict(color="#9ece6a", width=1.2),
        ))
        fig_mem.add_trace(go.Scatter(
            x=t, y=df_display["alloc_mem"],
            name="Allocated MEM",
            line=dict(color="#e0af68", width=2, dash="dash"),
        ))
        # MEM violations
        mem_viol_idx = df_display[df_display["sla_viol_mem"]==1].index
        if len(mem_viol_idx):
            fig_mem.add_trace(go.Scatter(
                x=mem_viol_idx,
                y=df_display.loc[mem_viol_idx, "actual_mem"],
                mode="markers",
                name="MEM Violation",
                marker=dict(color="#ff4444", size=5, symbol="x"),
            ))
        fig_mem.update_layout(
            height=280,
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1d27",
            font=dict(color="#a9b1d6"),
            legend=dict(bgcolor="#1a1d27"),
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis_title="Tick",
            yaxis_title="MEM Utilisation",
        )
        fig_mem.update_xaxes(gridcolor="#2d3147")
        fig_mem.update_yaxes(gridcolor="#2d3147")
        st.plotly_chart(fig_mem, use_container_width=True)
        st.caption(
            f"MEM SLA violation: "
            f"{float(df['sla_viol_mem'].mean())*100:.2f}%")
        
# ── Charts row 2: VM Activity + AWS Cost ──────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("## 🖥️ VM Activity")

    fig2 = go.Figure()

    if "vm_count" in df_display.columns:
        vm_vals = df_display["vm_count"].values

        # Step plot so flat sections show clearly
        fig2.add_trace(go.Scatter(
            x=t, y=vm_vals,
            name="Active VMs",
            mode="lines",
            line=dict(color="#9ece6a", width=2, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(158,206,106,0.15)",
        ))

        # Scale UP markers
        if "scaled_up" in df_display.columns:
            up_idx = df_display[df_display["scaled_up"] == True].index
            if len(up_idx):
                fig2.add_trace(go.Scatter(
                    x=up_idx,
                    y=df_display.loc[up_idx, "vm_count"] + 0.05,
                    mode="markers",
                    name="Scale Up ↑",
                    marker=dict(
                        color="#9ece6a", size=12,
                        symbol="triangle-up",
                        line=dict(color="#ffffff", width=1)
                    ),
                ))

        # Scale DOWN markers
        if "scaled_down" in df_display.columns:
            dn_idx = df_display[df_display["scaled_down"] == True].index
            if len(dn_idx):
                fig2.add_trace(go.Scatter(
                    x=dn_idx,
                    y=df_display.loc[dn_idx, "vm_count"] + 0.05,
                    mode="markers",
                    name="Scale Down ↓",
                    marker=dict(
                        color="#f7768e", size=12,
                        symbol="triangle-down",
                        line=dict(color="#ffffff", width=1)
                    ),
                ))

    fig2.update_layout(
        height=320,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#a9b1d6"),
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3147"),
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis_title="Tick",
        yaxis_title="VM Count",
        yaxis=dict(
            tickmode="linear", tick0=0, dtick=1,
            range=[0, df_display["vm_count"].max() + 0.5
                   if "vm_count" in df_display.columns else 4]
        ),
    )
    fig2.update_xaxes(gridcolor="#2d3147")
    fig2.update_yaxes(gridcolor="#2d3147")
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    st.markdown("## 💰 AWS Cost Over Time")

    fig3 = go.Figure()

    if "aws_total_cost" in df_display.columns:
        cum_total = df_display["aws_total_cost"].cumsum()
        cum_prov  = df_display["provisioning_cost"].cumsum() \
                    if "provisioning_cost" in df_display.columns \
                    else cum_total
        cum_waste = df_display["waste_cost"].cumsum() \
                    if "waste_cost" in df_display.columns \
                    else pd.Series(np.zeros(len(df_display)))

        fig3.add_trace(go.Scatter(
            x=t, y=cum_total,
            name="Total Cost",
            fill="tozeroy",
            fillcolor="rgba(224,175,104,0.20)",
            line=dict(color="#e0af68", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=t, y=cum_prov,
            name="Provisioning",
            line=dict(color="#7aa2f7", width=1.5, dash="dash"),
        ))
        fig3.add_trace(go.Scatter(
            x=t, y=cum_waste,
            name="Wasted Resources",
            fill="tozeroy",
            fillcolor="rgba(247,118,142,0.20)",
            line=dict(color="#f7768e", width=1.5),
        ))

    fig3.update_layout(
        height=320,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#a9b1d6"),
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3147"),
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis_title="Tick",
        yaxis_title="Cumulative Cost ($)",
    )
    fig3.update_xaxes(gridcolor="#2d3147")
    fig3.update_yaxes(gridcolor="#2d3147")
    st.plotly_chart(fig3, use_container_width=True)

# ── Charts row 3: Tier distribution + Over/Under prov ────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("## 📦 VM Tier Distribution")

    if "tier" in df.columns:
        tier_counts = df["tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_counts["pct"]  = tier_counts["count"] / tier_counts["count"].sum() * 100

        colors = {
            "XSMALL": "#7aa2f7",
            "SMALL" : "#9ece6a",
            "MEDIUM": "#e0af68",
            "LARGE" : "#f7768e",
            "XLARGE": "#bb9af7",
        }
        tier_colors = [colors.get(t, "#a9b1d6")
                       for t in tier_counts["tier"]]

        fig4 = go.Figure(go.Pie(
            labels=[f"{r['tier']}<br>{r['pct']:.1f}%"
                    for _, r in tier_counts.iterrows()],
            values=tier_counts["count"],
            marker_colors=tier_colors,
            hole=0.45,
            textinfo="label",
            textfont=dict(size=12, color="#c8d3f5"),
            insidetextorientation="radial",
        ))
        fig4.update_layout(
            height=320,
            paper_bgcolor="#0f1117",
            font=dict(color="#a9b1d6"),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(
                bgcolor="#1a1d27", bordercolor="#2d3147",
                font=dict(size=11)
            ),
        )
        st.plotly_chart(fig4, use_container_width=True)

with col_r2:
    st.markdown("## ⚖️ Over vs Under Provisioning")

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=t, y=df_display["over_prov"],
        name="Over-prov (cost×0.10)",
        fill="tozeroy",
        fillcolor="rgba(158,206,106,0.25)",
        line=dict(color="#9ece6a", width=1.2),
    ))
    fig5.add_trace(go.Scatter(
        x=t, y=-df_display["under_prov"],
        name="Under-prov (cost×0.50)",
        fill="tozeroy",
        fillcolor="rgba(247,118,142,0.25)",
        line=dict(color="#f7768e", width=1.2),
    ))
    fig5.add_hline(y=0, line_color="#a9b1d6", line_width=0.8)

    fig5.update_layout(
        height=320,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#a9b1d6"),
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3147"),
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis_title="Tick",
        yaxis_title="Provisioning Delta",
    )
    fig5.update_xaxes(gridcolor="#2d3147")
    fig5.update_yaxes(gridcolor="#2d3147")
    st.plotly_chart(fig5, use_container_width=True)

# ── Strategy comparison table ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Strategy Comparison (from notebook Table 4)")
st.markdown("*Green = best value in column. CQR achieves best SLA coverage.*")

comparison_data = {
    "Method"      : ["CQR Q95 (Ours)", "Uncalibrated Q95",
                     "Buffer 20%", "Buffer 10%", "Q50 Point"],
     "SLA Viol %"  : [3.96, 5.29, 24.14, 33.16, 44.73], 
    "Coverage %"  : [96.04, 94.71, 75.86, 66.84, 55.27],
    "Over-prov"   : [0.0188, 0.0183, 0.0068, 0.0051, 0.0037],
    "Under-prov"  : [0.0004, 0.0005, 0.0024, 0.0031, 0.0041],
    "Cost"        : [0.002095, 0.002059, 0.001885, 0.002046, 0.002396],
}
df_comp = pd.DataFrame(comparison_data)

def highlight_table(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    # Lower is better
    for col in ["SLA Viol %", "Under-prov"]:
        if col in df.columns:
            best = df[col].min()
            styles.loc[df[col] == best, col] = \
                "background-color:#1a3d2b; color:#4ade80; font-weight:bold"

    # Higher is better
    for col in ["Coverage %"]:
        if col in df.columns:
            best = df[col].max()
            styles.loc[df[col] == best, col] = \
                "background-color:#1a3d2b; color:#4ade80; font-weight:bold"

    # Highlight CQR row lightly
    styles.loc[0, "Method"] = \
        "background-color:#1a2d4d; color:#7aa2f7; font-weight:bold"

    return styles

styled = (df_comp.style
          .apply(highlight_table, axis=None)
          .format({
              "SLA Viol %" : "{:.2f}",
              "Coverage %" : "{:.2f}",
              "Over-prov"  : "{:.4f}",
              "Under-prov" : "{:.4f}",
              "Cost"       : "{:.6f}",
          }))

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Simulation stats ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔢 Full Simulation Statistics")

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("**Dataset**")
    st.markdown("Google Cluster 2019")
    st.markdown(f"Test ticks: `{len(df):,}`")
    st.markdown(f"Tick size: `5 min`")
    st.markdown(f"Horizon: `2 ticks (10 min)`")

with s2:
    st.markdown("**Model**")
    st.markdown("LightGBM Q95")
    st.markdown(f"Features: `47`")
    st.markdown(f"q_hat: `{q_hats['cpu']:.6f}`")
    st.markdown(f"Alpha: `0.95`")

with s3:
    st.markdown("**Scaling**")
    up   = int(df["scaled_up"].sum()  if "scaled_up"   in df.columns else 0)
    down = int(df["scaled_down"].sum() if "scaled_down" in df.columns else 0)
    st.markdown(f"Scale up events: `{up}`")
    st.markdown(f"Scale down events: `{down}`")
    st.markdown(f"Max VMs: `{int(df['vm_count'].max()) if 'vm_count' in df.columns else 'N/A'}`")
    st.markdown(f"Min VMs: `{int(df['vm_count'].min()) if 'vm_count' in df.columns else 'N/A'}`")

with s4:
    st.markdown("**AWS Cost**")
    if "aws_total_cost" in df.columns:
        sim_hours = len(df) * 5 / 60
        st.markdown(f"Total: `${df['aws_total_cost'].sum():.2f}`")
        st.markdown(f"Per hour: `${df['aws_total_cost'].sum()/sim_hours:.4f}`")
        st.markdown(f"Savings vs Buffer-20%: `16.7%`")
    st.markdown(f"Region: `ap-south-1`")

# ── Raw data ──────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown("---")
    st.markdown("## 🗂️ Raw Data (first 100 rows)")
    st.dataframe(df_display.head(100), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Cloud Resource Allocation using LGBM + CQR** · "
    "Google Cluster 2019 · Final Year Project"
)

if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
