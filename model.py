import math
import torch
import torch.nn as nn

POSE_FEATURES = 201  # 25 body×3 + 21 left_hand×3 + 21 right_hand×3


# ── Phase 1: BiLSTM ───────────────────────────────────────────────────────────

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

    def forward(self, x, src_key_padding_mask=None):
        out, _ = self.lstm(x)
        return self.fc(out)  # (batch, T, vocab_size)


# ── Phase 2: Transformer Encoder ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerSignModel(nn.Module):
    def __init__(self, in_size=POSE_FEATURES, d_model=256, nhead=4,
                 num_layers=4, dim_feedforward=1024, dropout=0.1, out_size=None):
        super().__init__()
        self.input_proj = nn.Linear(in_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN: more stable CTC training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, out_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)           # (batch, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.fc(x)                # (batch, T, vocab_size)
