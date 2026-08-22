import sys
import tempfile
import json
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to sys.path so we can import src and config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CFG
# Disable console spam in streamlit
CFG.DEBUG_STRIKES = False

from src.result_manager import ResultManager
from src.video_processor import VideoProcessor

# Configure page
st.set_page_config(
    page_title="Boxing AI Performance Analyzer",
    page_icon="🥊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stProgress .st-bo { background-color: #f63366; }
    .reportview-container { background: #0e1117; }
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def load_data(result_dir: Path):
    """Loads exported data from the result manager."""
    data = {}
    
    # JSON Summary
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data['metadata'] = json.load(f)
            
    stats_path = result_dir / "fight_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            data['stats'] = json.load(f)
            
    # CSVs
    events_csv = result_dir / "events.csv"
    if events_csv.exists() and events_csv.stat().st_size > 0:
        try:
            data['events'] = pd.read_csv(events_csv)
        except pd.errors.EmptyDataError:
            data['events'] = pd.DataFrame()
    else:
        data['events'] = pd.DataFrame()
        
    movement_csv = result_dir / "movement.csv"
    if movement_csv.exists() and movement_csv.stat().st_size > 0:
        try:
            data['movement'] = pd.read_csv(movement_csv)
        except pd.errors.EmptyDataError:
            data['movement'] = pd.DataFrame()
    else:
        data['movement'] = pd.DataFrame()
        
    data['video'] = result_dir / "boxing_analysis.mp4"
    return data

def main():
    st.title("🥊 BOXING AI PERFORMANCE ANALYZER")
    st.markdown("### Computer Vision Fight Analysis")
    st.markdown("An advanced AI tool utilizing pose estimation and heuristic action detection to analyze boxing performance. Upload a video to generate fight statistics, movement mapping, and strike classification.")

    st.markdown("---")

    # --- VIDEO INPUT ---
    st.header("1. Upload Video")
    uploaded_file = st.file_uploader("Upload a short fight clip (MP4 format)", type=["mp4", "mov", "avi"])

    # Provide an option to process max frames to prevent long processing in UI
    max_frames = st.slider("Max Frames to Process (Lower for faster testing)", min_value=100, max_value=2000, value=300, step=100)

    if uploaded_file is not None:
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(uploaded_file)
            
        with col2:
            st.info(f"Loaded: {uploaded_file.name} ({(uploaded_file.size/1e6):.2f} MB)")
            
            if st.button("Analyze Fight", type="primary"):
                # We need a real file path for cv2.VideoCapture
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)
                
                # --- PROCESSING PIPELINE ---
                st.subheader("Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing AI models...")
                
                try:
                    rm = ResultManager(uploaded_file.name)
                    processor = VideoProcessor(rm)
                    
                    # Instead of blocking entirely, we would ideally hook into VideoProcessor's frame loop, 
                    # but for this MVP we just run it directly. Streamlit will show the running indicator on top right.
                    status_text.text(f"Running full analytical stack (Tracker -> Pose -> Temporal -> Actions) over max {max_frames} frames... This may take a few minutes.")
                    
                    # Run it
                    processor.process_video(tmp_path, max_frames=max_frames)
                    
                    status_text.text("Analysis complete! Rendering dashboard...")
                    progress_bar.progress(100)
                    
                    # Save the result path to session state so it persists on rerenders
                    st.session_state['result_dir'] = str(rm.output_dir)
                    
                except Exception as e:
                    st.error(f"An error occurred during processing.")
                    st.error(str(e))
                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()

    st.markdown("---")
    
    # --- RESULTS DASHBOARD ---
    if 'result_dir' in st.session_state:
        st.header("2. Fight Results")
        result_dir = Path(st.session_state['result_dir'])
        data = load_data(result_dir)
        
        stats = data.get('stats', {}).get("fighters", {})
        
        if not stats:
            st.warning("No fighter data could be extracted. The video may not contain clear detections.")
            return

        fids = list(stats.keys())
        # Fallback names
        f1_id = fids[0] if len(fids) > 0 else 1
        f2_id = fids[1] if len(fids) > 1 else 2
        
        f1_stats = stats.get(str(f1_id), stats.get(f1_id, {}))
        f2_stats = stats.get(str(f2_id), stats.get(f2_id, {}))

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<div class='metric-card'><h3 style='color: #ff6464;'>Fighter {f1_id} (Blue)</h3></div>", unsafe_allow_html=True)
            st.metric("Total Punches", f1_stats.get("total_punches", 0))
            st.metric("Possible Landed", f1_stats.get("possible_landed", 0))
            st.metric("Defensive Blocks", f1_stats.get("blocks", 0))
            st.metric("Activity Score", f"{f1_stats.get('activity_score', 0):.1f}")
            st.metric("Stance", f1_stats.get("stance", "UNKNOWN"))

        with col2:
            st.markdown(f"<div class='metric-card'><h3 style='color: #6464ff;'>Fighter {f2_id} (Red)</h3></div>", unsafe_allow_html=True)
            st.metric("Total Punches", f2_stats.get("total_punches", 0))
            st.metric("Possible Landed", f2_stats.get("possible_landed", 0))
            st.metric("Defensive Blocks", f2_stats.get("blocks", 0))
            st.metric("Activity Score", f"{f2_stats.get('activity_score', 0):.1f}")
            st.metric("Stance", f2_stats.get("stance", "UNKNOWN"))

        st.caption("⚠️ Note: 'Possible Landed' and other metrics are heuristic estimates based on pose intersection. They are NOT ground truth.")

        st.markdown("---")
        
        # --- CHARTS ---
        st.header("3. Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Strike Distribution")
            # Create a dataframe for bar chart
            punch_data = []
            for fid, f_stats in [(f1_id, f1_stats), (f2_id, f2_stats)]:
                if f_stats:
                    punch_data.extend([
                        {"Fighter": f"Fighter {fid}", "Type": "Jabs", "Count": f_stats.get("jabs", 0)},
                        {"Fighter": f"Fighter {fid}", "Type": "Crosses", "Count": f_stats.get("crosses", 0)},
                        {"Fighter": f"Fighter {fid}", "Type": "Hooks", "Count": f_stats.get("hooks", 0)},
                        {"Fighter": f"Fighter {fid}", "Type": "Uppercuts", "Count": f_stats.get("uppercuts", 0)}
                    ])
            if punch_data:
                df_punches = pd.DataFrame(punch_data)
                fig1 = px.bar(df_punches, x="Fighter", y="Count", color="Type", barmode="group")
                st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            st.subheader("Movement & Control")
            mov_data = []
            for fid, f_stats in [(f1_id, f1_stats), (f2_id, f2_stats)]:
                if f_stats:
                    mov_data.extend([
                        {"Fighter": f"Fighter {fid}", "State": "Advancing", "Pct": f_stats.get("time_advancing_pct", 0)},
                        {"Fighter": f"Fighter {fid}", "State": "Retreating", "Pct": f_stats.get("time_retreating_pct", 0)},
                        {"Fighter": f"Fighter {fid}", "State": "Stationary", "Pct": f_stats.get("time_stationary_pct", 0)}
                    ])
            if mov_data:
                df_mov = pd.DataFrame(mov_data)
                fig2 = px.bar(df_mov, x="Pct", y="Fighter", color="State", orientation='h', barmode='stack')
                st.plotly_chart(fig2, use_container_width=True)

        # Timeline
        events_df = data.get("events")
        if events_df is not None and not events_df.empty:
            st.subheader("Event Timeline")
            # Plot strikes and defenses over time
            fig3 = px.scatter(events_df, x="frame_number", y="fighter_id", 
                              color="action", symbol="category",
                              hover_data=["confidence", "target_zone", "event_type"],
                              title="Events across Video Timeline")
            # Make Y axis discrete
            fig3.update_yaxes(type='category', title="Fighter ID")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        
        # --- VIDEO PLAYBACK ---
        st.header("4. Processed Video")
        video_path = data.get('video')
        if video_path and video_path.exists():
            # Browsers require H264 encoded MP4s to play back natively. 
            # If OpenCV wrote using 'mp4v' or another codec, it might not play without conversion.
            # We will attempt to play it directly.
            st.video(str(video_path))
        else:
            st.warning("Processed video file not found.")

        # --- DOWNLOADS ---
        st.header("5. Downloads")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        csv_path = result_dir / "fight_stats.csv"
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                dl_col1.download_button(
                    label="📥 Download Stats CSV",
                    data=f.read(),
                    file_name="fight_stats.csv",
                    mime="text/csv"
                )
                
        json_path = result_dir / "fight_stats.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                dl_col2.download_button(
                    label="📥 Download Stats JSON",
                    data=f.read(),
                    file_name="fight_stats.json",
                    mime="application/json"
                )
                
        if video_path and video_path.exists():
            with open(video_path, 'rb') as f:
                dl_col3.download_button(
                    label="📥 Download Video",
                    data=f.read(),
                    file_name="boxing_analysis.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()
