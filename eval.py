"""
Evaluate a saved checkpoint on the How2Sign validation split.

Usage:
    python eval.py \
        --checkpoint checkpoints/model_ctc_primary.pth \
        --keypoints  /data/how2sign/ \
        --csv        /data/how2sign/train_labels.csv \
        --beam_width 10
"""
import argparse, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from dataset import How2SignDataset, SignDataset
from vocab import Vocab
from model import TransformerSignModel, SignModel, POSE_FEATURES
from train import (evaluate, collate_fn, make_padding_mask,
                   ctc_decode, ctc_beam_search, cer, wer,
                   attn_greedy_decode)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--keypoints",   default=None)
    p.add_argument("--csv",         default=None)
    p.add_argument("--fake",        default="dataset")
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--beam_width",  type=int,   default=10,
                   help="CTC beam width (1=greedy, 10=beam search)")
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--samples",     type=int,   default=None,
                   help="Limit eval to first N val samples (None=all)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Checkpoint epoch: {ckpt['epoch']}  "
          f"saved CER: {ckpt.get('cer', float('inf')):.4f}")

    vocab = Vocab()
    vocab.w2i = ckpt["vocab"]
    vocab.i2w = {v: k for k, v in vocab.w2i.items()}
    vocab.idx = max(vocab.w2i.values()) + 1
    print(f"Vocab size: {len(vocab)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = TransformerSignModel(
        POSE_FEATURES, d_model=256, nhead=4,
        num_layers=4, dim_feedforward=1024,
        dropout=0.1, out_size=len(vocab)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.keypoints and args.csv:
        ds = How2SignDataset(args.keypoints, args.csv)
        print(f"How2Sign dataset: {len(ds)} clips")
    else:
        ds = SignDataset(args.fake)
        print(f"Fake dataset: {len(ds)} samples")

    n_val  = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val],
                             generator=torch.Generator().manual_seed(42))

    if "norm_mean" in ckpt:
        ds.set_norm_stats(ckpt["norm_mean"].to("cpu"), ckpt["norm_std"].to("cpu"))

    if args.samples:
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.samples, len(val_ds))))

    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        collate_fn=collate_fn, shuffle=False)
    print(f"Evaluating on {len(val_ds)} samples  |  beam_width={args.beam_width}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    import time
    t0 = time.time()
    ctc_cer, val_wer, attn_cer = evaluate(model, val_dl, vocab, device, args.beam_width)
    elapsed = time.time() - t0

    print(f"\n{'─'*50}")
    decode_label = f"beam={args.beam_width}" if args.beam_width > 1 else "greedy"
    print(f"CTC-CER  ({decode_label}): {ctc_cer:.4f}")
    print(f"WER      ({decode_label}): {val_wer:.4f}")
    if attn_cer is not None:
        print(f"ATT-CER  (greedy):        {attn_cer:.4f}")
    print(f"Eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
