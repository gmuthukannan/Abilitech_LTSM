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

class Conv1DSubsampler(nn.Module):
    """
    2x temporal downsampling via stride-2 Conv1D.
    Reduces T/L ratio to limit CTC collapse, and learns local temporal patterns.
    """
    def __init__(self, in_size, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_size, d_model, kernel_size=3, stride=2, padding=1)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, T, in_size)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (batch, T//2, d_model)
        return self.proj(self.drop(self.act(x)))

    def output_lengths(self, input_lengths):
        """Output lengths after stride-2 conv: ceil(T / 2)."""
        return (input_lengths + 1) // 2


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
        self.subsample   = Conv1DSubsampler(in_size, d_model, dropout)
        self.pos_enc     = PositionalEncoding(d_model, dropout)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN: more stable CTC training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.fc      = nn.Linear(d_model, out_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.subsample(x)      # (batch, T//2, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.fc(x)          # (batch, T//2, vocab_size)

    def subsampled_lengths(self, input_lengths):
        return self.subsample.output_lengths(input_lengths)
