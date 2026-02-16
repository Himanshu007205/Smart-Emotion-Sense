import sys
import os
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Fix import path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.record_video import record_video
from src.analyze_landmarks import analyze_stress
from src.analyze_regions import analyze_regional_stress
from src.visualize_stress import visualize_stress
from src.analyze_hrv_esp8266 import analyze_hrv_esp8266

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Facial + HRV Stress Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- SIDEBAR NAVIGATION ----
with st.sidebar:
    st.header("🎥 Control Panel")
    st.subheader("1️⃣ Record or Load Video")
    record_btn = st.button("🔴 Record (10s)", use_container_width=True)
    uploaded_file = st.file_uploader("📁 Upload video file", type=['mp4', 'avi', 'mov'])
    analyze_btn = st.button("🔬 Analyze Facial Stress", use_container_width=True)
    hrv_btn = st.button("💓 Measure HRV (ECG via ESP8266)", use_container_width=True)
    visualize_btn = st.button("🎨 Show Heatmap Visualization", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Results", use_container_width=True)

    st.markdown("---")
    st.subheader("ℹ️ System Status")

    if 'video_path' in st.session_state and st.session_state.video_path:
        st.write(f"📹 Video: {os.path.basename(st.session_state.video_path)}")
    else:
        st.write("📹 No video loaded")

    if 'landmark_results' in st.session_state and st.session_state.landmark_results:
        st.write("✅ Facial Analysis Ready")
    else:
        st.write("⏳ Facial Analysis Pending")

    if 'hrv_results' in st.session_state and st.session_state.hrv_results:
        st.write("💓 HRV Data Ready")
    else:
        st.write("⏳ No HRV Data Yet")

# ---- SESSION STATE INIT ----
for key in ['video_path', 'landmark_results', 'regional_results', 'hrv_results']:
    if key not in st.session_state:
        st.session_state[key] = None

# ---- VIDEO HANDLING ----
if record_btn:
    with st.spinner("Recording video..."):
        result = record_video(duration=10, fps=20)
        if result:
            st.session_state.video_path = result["output_path"]
            st.success(f"✅ Recorded {result['frames_recorded']} frames")
        else:
            st.error("❌ Recording failed")

if uploaded_file:
    video_path = f"data/recordings/uploaded_{uploaded_file.name}"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.video_path = video_path
    st.success(f"✅ Uploaded {uploaded_file.name}")

# ---- FACIAL STRESS ANALYSIS ----
if analyze_btn:
    if not st.session_state.video_path or not os.path.exists(st.session_state.video_path):
        st.error("❌ No video file selected!")
    else:
        with st.spinner("Analyzing facial stress..."):
            st.session_state.landmark_results = analyze_stress(st.session_state.video_path)
            st.session_state.regional_results = analyze_regional_stress(st.session_state.video_path)
        if st.session_state.landmark_results and st.session_state.regional_results:
            st.success("✅ Facial analysis complete!")
        else:
            st.error("⚠️ No face detected or analysis failed.")

# ---- HRV (ECG) ANALYSIS ----
if hrv_btn:
    with st.spinner("Collecting ECG data from ESP8266..."):
        try:
            facial_data = st.session_state.landmark_results or None
            result = analyze_hrv_esp8266(
                port="COM3", duration=20, live_plot=False, facial_result=facial_data
            )
            if result:
                st.session_state.hrv_results = result
                st.success("✅ HRV (and Combined) Analysis Complete!")
            else:
                st.error("❌ HRV capture failed.")
        except Exception as e:
            st.error(f"⚠️ Could not read HRV data: {e}")

# ---- VISUALIZATION ----
if visualize_btn:
    if not st.session_state.video_path or not os.path.exists(st.session_state.video_path):
        st.error("❌ No video file available for visualization!")
    else:
        visualize_stress(st.session_state.video_path)

# ---- CLEAR SESSION ----
if clear_btn:
    for key in ['video_path', 'landmark_results', 'regional_results', 'hrv_results']:
        st.session_state[key] = None
    st.rerun()

# ---- MAIN DASHBOARD LAYOUT ----
st.title("🧠 Facial + HRV Stress Analysis Dashboard")
st.markdown("---")

# ========== FACIAL ANALYSIS DISPLAY ==========
if st.session_state.landmark_results and st.session_state.regional_results:
    landmark_res = st.session_state.landmark_results
    regional_res = st.session_state.regional_results

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        level = landmark_res["stress_level"]
        st.markdown(f"### Facial Stress: **:red[{level}]**")
        st.metric("Normalized Score", f"{landmark_res['normalized_score']:.1%}")
        st.metric("Tension Index", f"{landmark_res['tension_index']:.2f}")

    st.markdown("---")

    # --- MOTION CHARTS ---
    left, right = st.columns(2)
    with left:
        st.subheader("📈 Facial Motion Timeline")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        timeline = landmark_res["motion_timeline"]
        ax1.plot(range(len(timeline)), timeline, color='dodgerblue')
        ax1.fill_between(range(len(timeline)), timeline, alpha=0.3, color='dodgerblue')
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Motion Magnitude")
        ax1.set_title("Facial Motion Over Time")
        st.pyplot(fig1)
        st.metric("Frames Analyzed", landmark_res["frames_analyzed"])
        st.metric("Mean Motion", f"{landmark_res['mean_motion']:.5f}")
        st.metric("Std Motion", f"{landmark_res['std_motion']:.5f}")

    with right:
        st.subheader("🎯 Regional Stress Scores")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        regions = list(regional_res["regional_scores"].keys())
        scores = [regional_res["regional_scores"][r] for r in regions]
        colors = ['green' if s < 1.5 else 'orange' if s < 3.0 else 'red' for s in scores]
        ax2.barh(regions, scores, color=colors)
        ax2.set_xlabel("Stress Score")
        ax2.set_title("Regional Stress Distribution")
        for i, score in enumerate(scores):
            ax2.text(score, i, f'{score:.2f}', va='center')
        st.pyplot(fig2)
        st.metric("Final Stress Index", f"{regional_res['final_index']:.2f}")
        st.metric("Regional Level", regional_res["stress_level"])

    st.markdown("---")

    st.subheader("📊 Regional Activity Timeline")
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    timelines = regional_res["regional_timelines"]
    colors_map = {'eyebrows': 'red', 'eyes': 'blue', 'mouth': 'orange', 'jaw': 'green'}
    for region, values in timelines.items():
        ax3.plot(range(len(values)), values, label=region.capitalize(), color=colors_map.get(region, 'gray'))
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Motion")
    ax3.set_title("Regional Activity Over Time")
    ax3.legend()
    st.pyplot(fig3)

    with st.expander("ℹ️ Regional Weights Explanation"):
        weights = regional_res["weights"]
        st.write("**How regions are weighted in final stress calculation:**")
        for region, weight in weights.items():
            st.write(f"- **{region.capitalize()}**: {weight:.0%}")
        st.info("Eyebrows and mouth are given more importance as they show more stress-related movements.")

# ========== HRV (ECG) DISPLAY ==========
if st.session_state.hrv_results:
    hrv = st.session_state.hrv_results
    st.markdown("---")
    st.subheader("💓 ECG-Based HRV Metrics")
    st.metric("Heart Rate", f"{hrv['heart_rate']:.1f} bpm")
    st.metric("SDNN", f"{hrv['sdnn']:.3f} s")
    st.metric("RMSSD", f"{hrv['rmssd']:.3f} s")
    st.metric("HRV Stress Level", hrv['stress_level'])

# ========== COMBINED METRIC DISPLAY ==========
if st.session_state.hrv_results and st.session_state.hrv_results.get("combined_score") is not None:
    combined = st.session_state.hrv_results["combined_score"]
    overall = st.session_state.hrv_results["combined_level"]

    st.markdown("---")
    st.subheader("🧩 Combined Facial + HRV Stress Level")
    st.metric("Combined Score", f"{combined:.2f}")
    st.metric("Overall Level", overall)

    # Optional bar visualization
    facial_score = st.session_state.landmark_results['normalized_score'] if st.session_state.landmark_results else 0
    hrv_score = 1 - st.session_state.hrv_results['hrv_score']
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ['Facial', 'HRV (Inverted)', 'Combined']
    values = [facial_score, hrv_score, combined]
    ax.bar(bars, values, color=['dodgerblue', 'orange', 'purple'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized Stress (0–1)")
    ax.set_title("Mind–Body Stress Comparison")
    st.pyplot(fig)

# ========== IF NOTHING RUN YET ==========
if not any([st.session_state.landmark_results, st.session_state.hrv_results]):
    st.info("👈 Use the sidebar to record or load a video, analyze facial stress, or measure HRV.")
    st.markdown("""
    ### 🚀 How to use:
    1. **Record or Upload Video**  
       Capture ~10s clip using your webcam or upload an existing one.
    2. **Analyze Facial Stress**  
       Uses MediaPipe FaceMesh for facial tension analysis.
    3. **Measure HRV (ECG)**  
       Collect ECG from AD8232 via ESP8266 to compute HRV-based stress.
    4. **View Combined Results**  
       Both signals are fused to estimate your overall stress level.
    """)

st.markdown("---")
st.markdown("🧠 **Facial + HRV Stress Analysis Dashboard | Powered by MediaPipe, OpenCV & ESP8266 AD8232**")
