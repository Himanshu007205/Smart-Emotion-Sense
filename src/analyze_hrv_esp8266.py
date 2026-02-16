import serial
import numpy as np
import time
import matplotlib.pyplot as plt

def analyze_hrv_esp8266(port="COM3", baudrate=115200, duration=5, live_plot=True, facial_result=None):
    print(f"🔌 Connecting to {port}...")
    ser = serial.Serial(port, baudrate, timeout=0.1)
    time.sleep(2)

    timestamps = []
    ecg_values = []
    start_time = time.time()

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Live ECG from AD8232 (ESP8266)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG Amplitude")
        line, = ax.plot([], [], lw=1)

    print(f"💓 Collecting ECG data for {duration} seconds...")
    while time.time() - start_time < duration:
        try:
            line_data = ser.readline().decode(errors='ignore').strip()
            if line_data.isdigit():
                val = int(line_data)
                ecg_values.append(val)
                timestamps.append(time.time() - start_time)

                if live_plot and len(ecg_values) % 10 == 0:
                    line.set_data(timestamps, ecg_values)
                    ax.set_xlim(max(0, timestamps[-1] - 10), timestamps[-1] + 1)
                    ax.set_ylim(min(ecg_values[-100:]) - 10, max(ecg_values[-100:]) + 10)
                    plt.pause(0.001)
        except:
            pass
        time.sleep(0.001)

    ser.close()
    print("✅ ECG data collection complete.")
    print(f"📊 Total samples: {len(ecg_values)}")

    if len(ecg_values) < 10:
        print("⚠️ Not enough ECG data for HRV analysis.")
        return {
            "data_points": len(ecg_values),
            "sampling_rate": 0,
            "heart_rate": 0,
            "sdnn": 0,
            "rmssd": 0,
            "hrv_score": 0,
            "stress_level": "UNKNOWN",
            "combined_score": None,
            "combined_level": "UNKNOWN"
        }
    t = np.array(timestamps)
    ecg = np.array(ecg_values)
    ecg = ecg - np.mean(ecg)

    fs = 1 / np.median(np.diff(t)) if len(t) > 2 else len(t) / duration
    print(f"📉 Estimated sampling rate: {fs:.2f} Hz")

    diff_vals = np.abs(np.diff(ecg))
    variability = np.mean(diff_vals)
    norm_var = np.clip(variability / (np.max(ecg) - np.min(ecg) + 1e-6), 0, 1)

    hrv_score = 1 - norm_var
    stress_level = (
        "LOW" if hrv_score > 0.6 else
        "MEDIUM" if hrv_score > 0.3 else
        "HIGH"
    )
    zero_crossings = np.sum(np.diff(np.sign(ecg)) != 0)
    est_hr = (zero_crossings / 2) / (duration / 60.0)

    print(f"\n💓 Approx HR: {est_hr:.1f} bpm")
    print(f"🧠 HRV Score: {hrv_score:.2f} → Stress: {stress_level}")

    if live_plot:
        plt.pause(2)
        plt.close(fig)
    combined_score = None
    combined_level = None
    if facial_result and "normalized_score" in facial_result:
        facial_score = facial_result["normalized_score"]
        hrv_stress_value = 1 - hrv_score
        combined_score = 0.6 * facial_score + 0.4 * hrv_stress_value

        if combined_score < 0.35:
            combined_level = "LOW"
        elif combined_score < 0.65:
            combined_level = "MEDIUM"
        else:
            combined_level = "HIGH"

        print("\n🧩 Combined Stress Evaluation:")
        print(f"Facial Score: {facial_score:.2f}")
        print(f"HRV Stress Value (inverted): {hrv_stress_value:.2f}")
        print(f"Final Combined Score: {combined_score:.2f}")
        print(f"➡️ Overall Stress Level: {combined_level}")
    return {
        "data_points": len(ecg),
        "sampling_rate": fs,
        "heart_rate": est_hr,
        "sdnn": np.std(np.diff(ecg)) if len(ecg) > 2 else 0,
        "rmssd": np.sqrt(np.mean(np.square(np.diff(ecg)))) if len(ecg) > 2 else 0,
        "hrv_score": hrv_score,
        "stress_level": stress_level,
        "combined_score": combined_score,
        "combined_level": combined_level
    }
if __name__ == "__main__":
    hrv_result = analyze_hrv_esp8266(port="COM3", duration=5, live_plot=True)