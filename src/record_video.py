import cv2
import time
import os

def record_video(output_path="data/recordings/user_clip.mp4", duration=10, fps=20):
    """
    Record video from webcam.
    Returns: dict with recording info or None if failed
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return None

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print(f"🎥 Recording for {duration} seconds...")
    start_time = time.time()
    frames_recorded = 0

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)    
        cv2.imshow('Recording... Press Q to stop early', frame)
        out.write(frame)
        frames_recorded += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    actual_duration = time.time() - start_time
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video saved to {output_path}")

    return {
        "output_path": output_path,
        "duration": actual_duration,
        "frames_recorded": frames_recorded,
        "fps": fps,
        "resolution": (frame_width, frame_height)
    }

if __name__ == "__main__":
    record_video()