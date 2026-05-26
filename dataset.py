import torch, os, numpy as np
from torch.utils.data import Dataset

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
