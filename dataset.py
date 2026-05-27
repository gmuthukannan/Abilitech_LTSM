import torch, os, glob, json, numpy as np
from torch.utils.data import Dataset

# ── Local fake-data dataset (pose.npy + text.txt per folder) ──────────────────

class SignDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = sorted([
            f for f in os.listdir(root)
            if not f.startswith('.') and os.path.isdir(os.path.join(root, f))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = os.path.join(self.root, self.samples[idx])
        pose = torch.tensor(np.load(os.path.join(p, "pose.npy")), dtype=torch.float32)
        with open(os.path.join(p, "text.txt")) as f:
            text = f.read().strip().upper()
        return pose, text


# ── How2Sign OpenPose dataset ─────────────────────────────────────────────────
# Expected layout after extracting train_2D_keypoints.tar.gz:
#   <keypoints_dir>/bfh_2d_front/openpose_output/json/<SENTENCE_ID>-rgb_front/*.json
# CSV must have columns: SENTENCE_ID, SENTENCE (tab-separated)

import pandas as pd

OPENPOSE_FEATURES = 201  # 25 body×3 + 21 left_hand×3 + 21 right_hand×3

def _load_openpose_clip(clip_dir):
    json_files = sorted(glob.glob(os.path.join(clip_dir, "*.json")))
    frames = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        people = data.get("people", [])
        if not people:
            frames.append(np.zeros(OPENPOSE_FEATURES, dtype=np.float32))
            continue
        p = people[0]
        pose  = np.array(p.get("pose_keypoints_2d",       [0.0] * 75), dtype=np.float32)
        lhand = np.array(p.get("hand_left_keypoints_2d",  [0.0] * 63), dtype=np.float32)
        rhand = np.array(p.get("hand_right_keypoints_2d", [0.0] * 63), dtype=np.float32)
        frames.append(np.concatenate([pose, lhand, rhand]))
    if not frames:
        return np.zeros((1, OPENPOSE_FEATURES), dtype=np.float32)
    return np.stack(frames, axis=0)


class How2SignDataset(Dataset):
    def __init__(self, keypoints_dir, csv_path, min_frames=10):
        """
        keypoints_dir: root of extracted tar.gz (contains bfh_2d_front/)
        csv_path:      How2Sign annotation CSV (tab-separated, has SENTENCE_ID + SENTENCE)
        """
        df = pd.read_csv(csv_path, sep="\t")
        json_root = os.path.join(
            keypoints_dir, "bfh_2d_front", "openpose_output", "json"
        )
        self.samples = []
        for _, row in df.iterrows():
            clip_dir = os.path.join(json_root, str(row["SENTENCE_NAME"]))
            if not os.path.isdir(clip_dir):
                continue
            n_frames = len(glob.glob(os.path.join(clip_dir, "*.json")))
            if n_frames < min_frames:
                continue
            sentence = str(row.get("SENTENCE", "")).strip().upper()
            if sentence:
                self.samples.append((clip_dir, sentence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_dir, text = self.samples[idx]
        pose = torch.tensor(_load_openpose_clip(clip_dir), dtype=torch.float32)
        return pose, text
