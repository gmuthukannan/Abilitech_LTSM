import torch.nn as nn

# 25 OpenPose body × 3 + 21 left_hand × 3 + 21 right_hand × 3
POSE_FEATURES = 201

class SignModel(nn.Module):
    def __init__(self, in_size=POSE_FEATURES, hidden=256, out_size=None):
        super().__init__()
        self.lstm = nn.LSTM(
            in_size, hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden * 2, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)  # (batch, T, vocab_size) — per-frame logits for CTC
