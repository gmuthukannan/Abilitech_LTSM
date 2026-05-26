"""
Extract MediaPipe Holistic keypoints from How2Sign videos.

How2Sign CSV columns used: VIDEO_ID, SENTENCE_ID, START_REALIGNED, END_REALIGNED, SENTENCE

Usage (on Brev H100 after downloading How2Sign):
    python extract_keypoints.py \
        --csv  /data/how2sign/train_labels_v1.csv \
        --videos /data/how2sign/train/rgb_front/ \
        --out  dataset/ \
        [--split val|test] \
        [--max_samples 1000]
"""
import argparse, os, cv2, numpy as np, pandas as pd
import mediapipe as mp
from tqdm import tqdm

# Must match model.py POSE_FEATURES
# 33 pose × 3 + 21 left_hand × 3 + 21 right_hand × 3 = 225
FEATURES = 225


def landmarks_to_array(landmarks, n_points):
    if landmarks is None:
        return np.zeros((n_points, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def frame_to_vector(results):
    pose  = landmarks_to_array(results.pose_landmarks, 33)        # 99
    lhand = landmarks_to_array(results.left_hand_landmarks, 21)   # 63
    rhand = landmarks_to_array(results.right_hand_landmarks, 21)  # 63
    return np.concatenate([pose.flatten(), lhand.flatten(), rhand.flatten()])  # 225


def extract_clip(video_path, start_sec, end_sec, holistic):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    vectors = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        vectors.append(frame_to_vector(results))

    cap.release()
    return np.stack(vectors, axis=0).astype(np.float32) if vectors else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         required=True,  help="How2Sign annotation CSV (tab-separated)")
    parser.add_argument("--videos",      required=True,  help="Directory containing .mp4 video files")
    parser.add_argument("--out",         default="dataset", help="Output directory for pose.npy + text.txt")
    parser.add_argument("--split",       default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Stop after N samples")
    parser.add_argument("--min_frames",  type=int, default=10,   help="Skip clips shorter than N frames")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, sep="\t")
    os.makedirs(args.out, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    processed = 0
    skipped = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {args.split}"):
            if args.max_samples and processed >= args.max_samples:
                break

            sentence_id = str(row["SENTENCE_ID"])
            out_dir = os.path.join(args.out, sentence_id)

            if os.path.exists(os.path.join(out_dir, "pose.npy")):
                processed += 1
                continue  # already extracted

            video_path = os.path.join(args.videos, str(row["VIDEO_ID"]) + ".mp4")
            if not os.path.exists(video_path):
                skipped += 1
                continue

            sentence = str(row.get("SENTENCE", "")).strip()
            if not sentence:
                skipped += 1
                continue

            try:
                pose = extract_clip(
                    video_path,
                    float(row["START_REALIGNED"]),
                    float(row["END_REALIGNED"]),
                    holistic
                )
            except Exception as e:
                print(f"\nError on {sentence_id}: {e}")
                skipped += 1
                continue

            if pose is None or len(pose) < args.min_frames:
                skipped += 1
                continue

            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "pose.npy"), pose)
            with open(os.path.join(out_dir, "text.txt"), "w") as f:
                f.write(sentence.upper())

            processed += 1

    print(f"\nDone: {processed} extracted, {skipped} skipped.")
    print(f"Output in: {args.out}")


if __name__ == "__main__":
    main()
