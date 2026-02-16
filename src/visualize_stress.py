import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from time import sleep
from scipy.ndimage import uniform_filter1d


# Configure page
st.set_page_config(
    page_title="Facial Stress Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-low {
        background-color: #10b981;
        color: white;
    }
    .status-medium {
        background-color: #f59e0b;
        color: white;
    }
    .status-high {
        background-color: #ef4444;
        color: white;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

REGIONS = {
    "eyebrows": list(range(55, 88)),
    "eyes": list(range(133, 159)) + list(range(362, 388)),
    "mouth": list(range(61, 88)) + list(range(308, 324)),
    "jaw": list(range(0, 17))
}

COLORS = {
    "low": (0, 255, 0),
    "medium": (0, 255, 255),
    "high": (0, 0, 255)
}

REGION_WEIGHTS = {
    "eyebrows": 0.35,
    "mouth": 0.25,
    "eyes": 0.25,
    "jaw": 0.15
}

def get_color_for_value(value):
    if value < 0.3:
        return COLORS["low"]
    elif value < 0.6:
        return COLORS["medium"]
    return COLORS["high"]

def create_gauge_chart(value, title):
    """Create a gauge chart for stress visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def create_region_bar_chart(scores):
    """Create bar chart for regional analysis"""
    regions = list(scores.keys())
    values = [scores[r] * 100 for r in regions]
    colors = ['#10b981' if v < 30 else '#f59e0b' if v < 60 else '#ef4444' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=regions,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Regional Stress Distribution",
        xaxis_title="Facial Region",
        yaxis_title="Stress Level (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 100], gridcolor='lightgray')
    )
    return fig

def visualize_stress(video_path):
    # Header
    st.markdown('<div class="main-header">🧠 Facial Stress Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time facial micro-expression analysis for stress detection</div>', unsafe_allow_html=True)
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Analysis Settings")
        st.markdown("""
        ### Color Legend
        - 🟢 **Green**: Low stress (0-30%)
        - 🟡 **Yellow**: Medium stress (30-60%)
        - 🔴 **Red**: High stress (60-100%)
        
        ### Key Regions
        - **Eyebrows** (35%): Major stress indicator
        - **Mouth** (25%): Tension and expression
        - **Eyes** (25%): Micro-movements
        - **Jaw** (15%): Muscle tension
        """)
        
        st.info("💡 The algorithm tracks subtle facial movements across multiple regions to estimate stress levels.")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    prev_landmarks = None
    region_motion = {r: [] for r in REGIONS}

    # Create layout for live analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎥 Live Video Feed")
        frame_window = st.empty()
    
    with col2:
        st.markdown("### 📈 Real-time Metrics")
        metrics_placeholder = st.empty()
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress = frame_count / total_frames if total_frames > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            bbox_min, bbox_max = landmarks.min(axis=0), landmarks.max(axis=0)
            landmarks = (landmarks - bbox_min) / (bbox_max - bbox_min + 1e-6)

            h, w, _ = frame.shape
            points_2d = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])

            if prev_landmarks is not None:
                diff = np.linalg.norm(landmarks - prev_landmarks, axis=1)
                for region, idxs in REGIONS.items():
                    region_motion[region].append(np.mean(diff[idxs]))

                # Draw colored landmarks
                for region, idxs in REGIONS.items():
                    vals = region_motion[region]
                    if len(vals) > 3:
                        smoothed = uniform_filter1d(vals[-5:], size=3)
                        normalized = np.clip(smoothed[-1] / (np.percentile(smoothed, 95) + 1e-6), 0, 1)
                        color = get_color_for_value(normalized)
                        for i in idxs:
                            x, y = points_2d[i]
                            cv2.circle(frame, (x, y), 2, color, -1)

                # Update real-time metrics
                if frame_count % 10 == 0:  # Update every 10 frames
                    current_scores = {}
                    for region, vals in region_motion.items():
                        if vals:
                            current_scores[region] = np.mean(vals[-10:]) if len(vals) > 10 else np.mean(vals)
                    
                    with metrics_placeholder.container():
                        for region, val in current_scores.items():
                            normalized_val = np.clip(val * 100, 0, 100)
                            st.metric(
                                label=f"{region.capitalize()}",
                                value=f"{normalized_val:.1f}%",
                                delta=None
                            )

            prev_landmarks = landmarks

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        sleep(0.03)

    cap.release()
    face_mesh.close()
    progress_bar.empty()
    status_text.empty()

    # Calculate final scores
    scores = {}
    for region, vals in region_motion.items():
        if vals:
            scores[region] = np.mean(uniform_filter1d(vals, size=5))

    # Normalize scores
    max_score = max(scores.values()) if scores else 1
    normalized_scores = {k: v/max_score for k, v in scores.items()}

    final_index = sum(REGION_WEIGHTS[region] * scores.get(region, 0) for region in REGIONS)
    normalized_final = np.clip(final_index / max_score, 0, 1)

    level = "LOW" if normalized_final < 0.3 else "MEDIUM" if normalized_final < 0.6 else "HIGH"
    level_class = f"status-{level.lower()}"

    # Results Section
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Main metrics row
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Stress Index</h3>
            <h1>{normalized_final*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Stress Level</h3>
            <div class="status-badge {level_class}">{level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Frames Analyzed</h3>
            <h1>{frame_count}</h1>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.markdown("---")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.plotly_chart(create_gauge_chart(normalized_final, "Stress Gauge"), use_container_width=True)
    
    with viz_col2:
        st.plotly_chart(create_region_bar_chart(normalized_scores), use_container_width=True)

    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 🔍 Detailed Regional Analysis")
    
    detail_cols = st.columns(4)
    for idx, (region, val) in enumerate(normalized_scores.items()):
        with detail_cols[idx]:
            weight = REGION_WEIGHTS[region]
            st.markdown(f"""
            <div class="info-box">
                <h4>{'💪' if region=='jaw' else '👁️' if region=='eyes' else '👄' if region=='mouth' else '🤨'} {region.capitalize()}</h4>
                <p><strong>Score:</strong> {val*100:.1f}%</p>
                <p><strong>Weight:</strong> {weight*100:.0f}%</p>
                <p><strong>Contribution:</strong> {val*weight*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # Interpretation
    st.markdown("---")
    st.markdown("### 💡 Interpretation")
    if level == "LOW":
        st.success("✅ **Low Stress Detected**: The subject appears calm with minimal facial tension. Micro-movements are within normal relaxed ranges.")
    elif level == "MEDIUM":
        st.warning("⚠️ **Moderate Stress Detected**: Notable facial micro-movements suggest some tension. Consider the context and environmental factors.")
    else:
        st.error("🚨 **High Stress Detected**: Significant facial tension and micro-movements observed. This may indicate elevated stress levels.")

# Main execution
if __name__ == "__main__":
    st.sidebar.header("📁 Video Input")
    video_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file is not None:
        # Save uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        if st.sidebar.button("🚀 Start Analysis", type="primary"):
            visualize_stress("temp_video.mp4")
    else:
        st.markdown('<div class="main-header">🧠 Facial Stress Analyzer</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload a video to begin analysis</div>', unsafe_allow_html=True)
        
        st.info("👈 Please upload a video file from the sidebar to start the facial stress analysis.")
        
        # Show example of what to expect
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🎯 Features
            - Real-time facial tracking
            - Multi-region analysis
            - Color-coded stress mapping
            - Weighted scoring system
            """)
        with col2:
            st.markdown("""
            ### 📊 Metrics
            - Overall stress index
            - Regional breakdowns
            - Visual gauges & charts
            - Frame-by-frame analysis
            """)
        with col3:
            st.markdown("""
            ### 🔬 Technology
            - MediaPipe Face Mesh
            - 468 facial landmarks
            - Motion tracking
            - Statistical smoothing
            """)