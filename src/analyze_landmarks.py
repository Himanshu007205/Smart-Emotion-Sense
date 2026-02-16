import cv2
import mediapipe as mp
import numpy as np

def analyze_stress(video_path="data/user_clip.mp4"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    prev_landmarks = None
    motion_values = []

    frame_count = 0

    print("🧠 Analyzing facial movements...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

            if prev_landmarks is not None:
                # Calculate movement magnitude per landmark
                movement = np.linalg.norm(landmarks - prev_landmarks, axis=1)
                motion_values.append(np.mean(movement))

            prev_landmarks = landmarks

        frame_count += 1

    cap.release()
    face_mesh.close()

    if not motion_values:
        print("⚠️ No face detected in most frames.")
        return

    motion_values = np.array(motion_values)
    mean_motion = np.mean(motion_values)
    std_motion = np.std(motion_values)
    tension_index = (mean_motion + std_motion) * 100  # scale up for easier interpretation

    # Normalize roughly to [0, 1]
    normalized = min(1.0, tension_index / 5.0)

    if normalized < 0.3:
        stress_level = "LOW"
    elif normalized < 0.6:
        stress_level = "MEDIUM"
    else:
        stress_level = "HIGH"

    print("\n📊 Facial Motion Stats:")
    print(f"Frames analyzed: {frame_count}")
    print(f"Mean motion: {mean_motion:.5f}")
    print(f"Std motion: {std_motion:.5f}")
    print(f"Tension index: {tension_index:.2f}")
    print(f"🧩 Estimated Stress Level: {stress_level}")

    return {
        "frames_analyzed": frame_count,
        "mean_motion": mean_motion,
        "std_motion": std_motion,
        "tension_index": tension_index,
        "normalized_score": normalized,
        "stress_level": stress_level,
        "motion_timeline": motion_values.tolist(),
    }


if __name__ == "__main__":
    analyze_stress()
