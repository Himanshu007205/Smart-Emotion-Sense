# 🧠 Smart Emotion and Stress Monitoring System  
**IoT + Edge AI Based Multimodal Stress Analysis**

---

## 📌 Overview

The **Smart Emotion and Stress Monitoring System** is a multimodal stress assessment platform that combines **facial micro-expression analysis** with **physiological signal analysis (ECG-based HRV)** to estimate a user’s stress level in real time.

The system integrates **Edge AI**, **IoT sensing**, and **computer vision** to provide a **non-invasive, explainable, and real-time stress monitoring solution** by fusing facial behavior and heart activity.

---

## 🧠 System Architecture

**Inputs**
- Webcam video (facial expressions)
- ECG signal from AD8232 via ESP8266

**Processing Pipeline**
1. Video capture
2. Facial landmark extraction (MediaPipe FaceMesh – 468 landmarks)
3. Region-wise facial motion analysis
4. ECG signal acquisition and HRV estimation
5. Multimodal stress score fusion
6. Visualization and interpretation

**Outputs**
- Stress level: **LOW / MEDIUM / HIGH**
- Regional stress contribution
- HRV metrics and combined stress score

---

## 📂 Project Structure

Emotion-Monitor/
│
├── src/ # Core analysis modules
│ ├── record_video.py
│ ├── analyze_landmarks.py
│ ├── analyze_regions.py
│ ├── analyze_hrv_esp8266.py
│ └── visualize_stress.py
│
├── dashboard/ # Streamlit dashboard
│ └── app.py
│
├── data/ # Data directory (no real data stored)
│ └── README.md
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

---

## 🔬 Core Modules

### Facial Stress Analysis
- Tracks motion across **468 facial landmarks**
- Computes facial tension using statistical motion analysis
- Produces a normalized facial stress score

### Regional Facial Analysis
Stress is computed independently for key facial regions:

| Region     | Weight |
|-----------|--------|
| Eyebrows  | 35% |
| Mouth     | 25% |
| Eyes      | 25% |
| Jaw       | 15% |

This enables **explainable stress estimation** rather than a black-box model.

### ECG-Based HRV Analysis
- ECG data collected via **ESP8266 + AD8232**
- HRV metrics computed:
  - Heart Rate
  - SDNN
  - RMSSD
- HRV variability mapped to stress level

### Multimodal Stress Fusion
Facial and HRV stress signals are combined as:

Combined Stress = 0.6 × Facial Stress + 0.4 × HRV Stress


This improves robustness by correlating **external facial cues** with **internal physiological response**.

---

## 📊 Dashboard & Visualization

The Streamlit dashboard provides:
- Video recording or upload
- Real-time facial stress analysis
- HRV measurement trigger
- Regional stress breakdown
- Combined stress visualization
- Clear LOW / MEDIUM / HIGH interpretation

---

## 🚀 How to Run

### Install Dependencies
``bash
pip install -r requirements.txt

Launch Dashboard
streamlit run dashboard/app.py

🎶Usage Flow

Record or upload a video

Run facial stress analysis

(Optional) Collect ECG via ESP8266

View combined stress results

🔐 Data & Privacy

No real datasets are included in this repository

Facial videos and ECG data are processed locally

Design follows privacy, ethical, and reproducibility best practices

🛠️ Tech Stack

Python

OpenCV

MediaPipe FaceMesh

NumPy / SciPy

Matplotlib

Streamlit

ESP8266 + AD8232 ECG Sensor

🎯 Applications

Mental health monitoring

Human–Computer Interaction

IoT-based healthcare systems

Stress-aware smart environments

Academic research and prototyping

⚠️ Disclaimer

This project is intended for educational and research purposes only
and is not a medical diagnostic device.
