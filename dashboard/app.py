"""
dashboard/app.py — Streamlit dashboard for Boxing AI Performance Analyzer.

Displays fight statistics, punch timelines, and movement heatmaps from the
pipeline's JSON and CSV outputs.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
import os

# Ensure we can import config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from config import CFG
except ImportError:
    st.error("Failed to import config. Ensure you run this from the project root.")
    st.stop()


# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Boxing AI Analyzer",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional CSS for better aesthetics
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    summary_path = CFG.SUMMARY_JSON
    stats_path = CFG.STATS_CSV

    if not summary_path.exists() or not stats_path.exists():
        return None, None

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    stats = pd.read_csv(stats_path)
    return summary, stats


summary, stats = load_data()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🥊 Boxing AI Performance Analyzer")

if summary is None or stats is None:
    st.warning(
        "No analysis data found! Please run the pipeline first:\n\n"
        "`python main.py --video data/fight.mp4`"
    )
    st.stop()

st.sidebar.header("Fight Summary")
st.sidebar.markdown(f"**Frames Processed:** {summary.get('total_frames_processed', 0)}")
fighters = summary.get("fighters", {})

if not fighters:
    st.info("No fighters detected in the video.")
    st.stop()

# Assign distinct colors to fighters for plots
COLORS = ["#00FF80", "#0080FF", "#FF8000", "#FF0080"]

# ---------------------------------------------------------------------------
# Summary Metrics
# ---------------------------------------------------------------------------
st.subheader("Fighter Overview")

cols = st.columns(len(fighters))

for i, (fid, fstats) in enumerate(fighters.items()):
    with cols[i]:
        color = COLORS[i % len(COLORS)]
        st.markdown(
            f"""
            <div style="border-top: 4px solid {color}; padding-top: 10px;">
                <h3>Fighter {fid}</h3>
                <p><b>Total Punches:</b> {fstats['total_punches']}</p>
                <p><b>Aggression Score:</b> {fstats['aggression_score']:.3f}</p>
                <p><b>Distance Moved:</b> {fstats['total_distance_px']:.0f} px</p>
                <p><b>Defense (Guards/Slips/Ducks):</b> {fstats['guards']} / {fstats['slips']} / {fstats['ducks']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Detailed Stats Charts
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🥊 Punches & Strikes", "🛡️ Defense & Clinch", "📍 Movement & Positioning"])

with tab1:
    st.subheader("Punch Breakdown")
    # Prepare data for punch type breakdown
    punch_data = []
    for fid, fstats in fighters.items():
        for ptype in ["jabs", "crosses", "hooks", "uppercuts"]:
            punch_data.append(
                {"Fighter": f"Fighter {fid}", "Type": ptype.capitalize(), "Count": fstats[ptype]}
            )
    
    if punch_data:
        df_punches = pd.DataFrame(punch_data)
        fig_punches = px.bar(
            df_punches, x="Fighter", y="Count", color="Type", barmode="group",
            title="Punches by Type",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_punches, use_container_width=True)
    
    # Timeline of punches
    st.subheader("Punch Timeline")
    # Filter stats to only frames where a strike was thrown
    strikes_df = stats[stats["strike_label"] != "NONE"].copy()
    if not strikes_df.empty:
        # Convert frame index to rough time if FPS is 25
        strikes_df["time_sec"] = strikes_df["frame_idx"] / CFG.OUTPUT_VIDEO_FPS
        strikes_df["Fighter"] = "Fighter " + strikes_df["track_id"].astype(str)
        
        fig_timeline = px.scatter(
            strikes_df, x="time_sec", y="Fighter", color="strike_label",
            title="Strikes Over Time",
            labels={"time_sec": "Time (seconds)", "strike_label": "Strike Type"},
            color_discrete_sequence=px.colors.qualitative.Set1,
            size_max=10
        )
        fig_timeline.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No punches detected.")

with tab2:
    st.subheader("Defense & Clinches")
    def_data = []
    for fid, fstats in fighters.items():
        for dtype in ["guards", "slips", "ducks", "clinches"]:
            def_data.append(
                {"Fighter": f"Fighter {fid}", "Type": dtype.capitalize(), "Count": fstats[dtype]}
            )
    
    if def_data:
        df_def = pd.DataFrame(def_data)
        fig_def = px.bar(
            df_def, x="Type", y="Count", color="Fighter", barmode="group",
            title="Defensive Actions",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_def, use_container_width=True)

with tab3:
    st.subheader("Movement Heatmap")
    st.markdown("Visualising where fighters spent most of their time in the frame.")
    
    heatmap_cols = st.columns(len(fighters))
    for i, fid in enumerate(fighters.keys()):
        with heatmap_cols[i]:
            f_stats = stats[stats["track_id"] == int(fid)]
            if not f_stats.empty:
                # 2D Histogram of center coordinates
                fig_heat = px.density_contour(
                    f_stats, x="center_x", y="center_y",
                    title=f"Fighter {fid} Positioning",
                    width=400, height=400
                )
                fig_heat.update_traces(contours_coloring="fill", contours_showlabels=True)
                # Invert Y axis to match image coordinates (origin top-left)
                fig_heat.update_yaxes(autorange="reversed")
                # Hide axes to focus on heatmap
                fig_heat.update_xaxes(showticklabels=False, title="")
                fig_heat.update_yaxes(showticklabels=False, title="")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info(f"No movement data for Fighter {fid}.")

st.divider()
st.markdown("Developed with Streamlit and Plotly for Boxing AI Performance Analyzer.")
