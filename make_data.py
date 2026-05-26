"""Generate fake samples for local pipeline testing (CPU, no real video needed)."""
import os, numpy as np
from model import POSE_FEATURES

SAMPLES = [
    ("sample_001", "HELLO WORLD"),
    ("sample_002", "HOW ARE YOU"),
    ("sample_003", "MY NAME IS"),
    ("sample_004", "THANK YOU"),
    ("sample_005", "GOOD MORNING"),
    ("sample_006", "PLEASE HELP ME"),
    ("sample_007", "I UNDERSTAND"),
    ("sample_008", "NICE TO MEET YOU"),
]

for name, label in SAMPLES:
    out = f"dataset/{name}"
    os.makedirs(out, exist_ok=True)
    T = np.random.randint(40, 80)  # random sequence length (frames)
    np.save(f"{out}/pose.npy", np.random.rand(T, POSE_FEATURES).astype("float32"))
    with open(f"{out}/text.txt", "w") as f:
        f.write(label)

print(f"Created {len(SAMPLES)} fake samples  (shape: T × {POSE_FEATURES})")
