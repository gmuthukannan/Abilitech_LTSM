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


# ── Phase 2: Transformer Encoder + Attention Decoder ─────────────────────────

class Conv1DSubsampler(nn.Module):
    """2x temporal downsampling via stride-2 Conv1D."""
    def __init__(self, in_size, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_size, d_model, kernel_size=3, stride=2, padding=1)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (batch, T//2, d_model)
        return self.proj(self.drop(self.act(x)))

    def output_lengths(self, input_lengths):
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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentionDecoder(nn.Module):
    """
    Autoregressive Transformer decoder with cross-attention to encoder output.
    Trained with teacher forcing; provides secondary loss signal alongside CTC
    to prevent CTC collapse.
    """
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        dec_layer    = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.fc      = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # tgt: (batch, L) token indices
        # memory: (batch, T, d_model) encoder output
        L = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=tgt.device, dtype=torch.bool
        )
        x = self.pos_enc(self.embed(tgt))
        x = self.decoder(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc(x)  # (batch, L, vocab_size)


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
            norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                              enable_nested_tensor=False)
        self.ctc_fc   = nn.Linear(d_model, out_size)
        self.attn_dec = AttentionDecoder(out_size, d_model, nhead,
                                         num_layers=2,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout)

    def encode(self, x, src_key_padding_mask=None):
        x = self.subsample(x)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def forward(self, x, src_key_padding_mask=None):
        """CTC path — used for evaluation/inference."""
        return self.ctc_fc(self.encode(x, src_key_padding_mask))

    def forward_train(self, x, tgt_in, src_key_padding_mask=None,
                      tgt_key_padding_mask=None):
        """Returns (ctc_logits, attn_logits) for hybrid CTC+Attention training."""
        memory      = self.encode(x, src_key_padding_mask)
        ctc_logits  = self.ctc_fc(memory)
        attn_logits = self.attn_dec(
            tgt_in, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return ctc_logits, attn_logits

    def subsampled_lengths(self, input_lengths):
        return self.subsample.output_lengths(input_lengths)
