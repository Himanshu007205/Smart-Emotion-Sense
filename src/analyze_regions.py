import cv2
import mediapipe as mp
import numpy as np
REGIONS = {
    "eyebrows": list(range(70, 105)),
    "eyes": list(range(130, 160)),
    "mouth": list(range(0, 17)) + list(range(61, 88)),
    "jaw": list(range(200, 250))
}
def analyze_regional_stress(video_path="data/user_clip.mp4"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    prev_landmarks = None
    region_motion = {r: [] for r in REGIONS}
    total_frames = 0

    print("🧠 Analyzing facial region activity...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in result.multi_face_landmarks[0].landmark])

            if prev_landmarks is not None:
                diff = np.linalg.norm(landmarks - prev_landmarks, axis=1)
                for region, indices in REGIONS.items():
                    region_motion[region].append(np.mean(diff[indices]))

            prev_landmarks = landmarks
            total_frames += 1

    cap.release()
    face_mesh.close()

    if total_frames == 0:
        print("⚠️ No face detected.")
        return

    print("\n📊 Regional Facial Activity Summary:")
    stress_components = {}

    for region, motions in region_motion.items():
        if motions:
            mean_motion = np.mean(motions)
            std_motion = np.std(motions)
            score = (mean_motion + std_motion) * 100
            stress_components[region] = score
            print(f"{region.capitalize():10s}: {score:.2f}")
    final_index = (
        0.35 * stress_components.get("eyebrows", 0) +
        0.25 * stress_components.get("mouth", 0) +
        0.25 * stress_components.get("eyes", 0) +
        0.15 * stress_components.get("jaw", 0)
    )

    normalized = min(1.0, final_index / 5.0)
    if normalized < 0.3:
        level = "LOW"
    elif normalized < 0.6:
        level = "MEDIUM"
    else:
        level = "HIGH"

    print(f"\n🧩 Final Stress Index: {final_index:.2f}")
    print(f"🧠 Estimated Stress Level: {level}")

    return {
        "regional_scores": stress_components,
        "final_index": final_index,
        "normalized": normalized,
        "stress_level": level,
        "weights": {
            "eyebrows": 0.35,
            "mouth": 0.25,
            "eyes": 0.25,
            "jaw": 0.15,
        },
        "regional_timelines": region_motion,
    }
if __name__ == "__main__":
    analyze_regional_stress()
