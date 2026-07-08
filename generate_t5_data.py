"""
Generate (ctc_prediction, target_sentence) pairs for T5 fine-tuning.

Two modes:
  --synthetic   Read sentences from CSV and corrupt them at a realistic
                CTC noise level. Avoids the train/val distribution mismatch
                (CTC predictions on the train set are nearly perfect because
                the model memorised those clips; on val they have 0.75 CER).
                Val pairs are always generated from real CTC predictions.

  (default)     Run CTC inference on training clips (original behaviour).

Usage:
    # Recommended — synthetic noise training data, real CTC val data
    python generate_t5_data.py \
        --checkpoint checkpoints/model_ctc_primary.pth \
        --keypoints  /data/how2sign/ \
        --csv        /data/how2sign/train_labels.csv \
        --synthetic

    # Original mode — real CTC predictions for both splits
    python generate_t5_data.py \
        --checkpoint checkpoints/model_ctc_primary.pth \
        --keypoints  /data/how2sign/ \
        --csv        /data/how2sign/train_labels.csv \
        --beam_width 1
"""
import argparse, csv, random, string, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pandas as pd
from dataset import How2SignDataset, SignDataset
from vocab import Vocab
from model import TransformerSignModel, POSE_FEATURES
from train import collate_fn, make_padding_mask, ctc_decode, ctc_beam_search

_CHARS = list(string.ascii_uppercase + ' ')


def corrupt(sentence, target_cer=0.75, n_copies=3):
    """
    Synthetically corrupt a clean sentence to simulate CTC output at
    approximately target_cer CER.  Returns n_copies independent corruptions.

    Calibration (per character):
      P(delete)     = 0.45
      P(substitute) = 0.20
      P(insert)     = 0.10
      P(keep)       = 0.25
    Expected CER = 0.45 + 0.20 + 0.10 ≈ 0.75
    """
    results = []
    for _ in range(n_copies):
        out = []
        for ch in sentence:
            r = random.random()
            if r < 0.45:                        # delete
                pass
            elif r < 0.65:                      # substitute
                out.append(random.choice(_CHARS))
            elif r < 0.75:                      # insert then keep
                out.append(random.choice(_CHARS))
                out.append(ch)
            else:                               # keep
                out.append(ch)
        results.append(''.join(out))
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--keypoints",   default=None)
    p.add_argument("--csv",         default=None)
    p.add_argument("--fake",        default="dataset")
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--beam_width",  type=int,   default=1)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--out_train",   default="t5_train.tsv")
    p.add_argument("--out_val",     default="t5_val.tsv")
    p.add_argument("--synthetic",   action="store_true",
                   help="Generate train pairs by corrupting clean sentences "
                        "(fixes train/val CER distribution mismatch)")
    p.add_argument("--noise_copies", type=int, default=3,
                   help="Corrupted versions per sentence in synthetic mode")
    return p.parse_args()


def run_inference(model, loader, vocab, device, beam_width):
    model.eval()
    decode_fn = (lambda lp: ctc_beam_search(lp, vocab, beam_width)) if beam_width > 1 \
                else (lambda lp: ctc_decode(lp, vocab))
    rows = []
    with torch.no_grad():
        for padded, texts, input_lengths in loader:
            padded        = padded.to(device)
            input_lengths = input_lengths.to(device)
            if hasattr(model, 'subsampled_lengths'):
                out_lengths = model.subsampled_lengths(input_lengths)
                sub_T = model.subsampled_lengths(input_lengths.new_tensor([padded.size(1)])).item()
                mask = make_padding_mask(out_lengths, sub_T, device)
            else:
                out_lengths = input_lengths
                mask = make_padding_mask(input_lengths, padded.size(1), device)
            memory = model.encode(padded, mask)
            lp = F.log_softmax(model.ctc_fc(memory), dim=-1)
            for i, text in enumerate(texts):
                pred = decode_fn(lp[i].cpu())
                rows.append((pred, text.upper()))
    return rows


def write_tsv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["input", "target"])
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Checkpoint epoch: {ckpt['epoch']}  CER: {ckpt.get('cer', '?'):.4f}")

    vocab = Vocab()
    vocab.w2i = ckpt["vocab"]
    vocab.i2w = {v: k for k, v in vocab.w2i.items()}
    vocab.idx = max(vocab.w2i.values()) + 1

    model = TransformerSignModel(
        POSE_FEATURES, d_model=256, nhead=4,
        num_layers=4, dim_feedforward=1024,
        dropout=0.0, out_size=len(vocab)
    ).to(device)
    model.load_state_dict(ckpt["model"])

    if args.keypoints and args.csv:
        ds = How2SignDataset(args.keypoints, args.csv)
    else:
        ds = SignDataset(args.fake)
    if "norm_mean" in ckpt:
        ds.set_norm_stats(ckpt["norm_mean"].cpu(), ckpt["norm_std"].cpu())

    n_val   = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # ── Val: always use real CTC predictions ─────────────────────────────────
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        collate_fn=collate_fn, shuffle=False)
    print("Running CTC inference on val split (real predictions)...")
    val_rows = run_inference(model, val_dl, vocab, device, args.beam_width)

    # ── Train: synthetic noise OR real CTC ───────────────────────────────────
    if args.synthetic:
        # Read all sentences from CSV, corrupt to simulate ~0.75 CER noise.
        # Bypasses the train/val distribution mismatch: the CTC model has
        # memorised training clips (low CER there) but sees 0.75 CER on val.
        if args.csv:
            df = pd.read_csv(args.csv, sep="\t")
            sentences = [str(r).strip().upper()
                         for r in df["SENTENCE"].dropna() if str(r).strip()]
        else:
            sentences = [tgt for _, tgt in val_rows]   # fallback: val targets

        train_rows = []
        random.seed(42)
        for sent in sentences:
            for noisy in corrupt(sent, n_copies=args.noise_copies):
                if noisy.strip():
                    train_rows.append((noisy, sent))
        random.shuffle(train_rows)
        print(f"Synthetic train: {len(train_rows)} pairs "
              f"({len(sentences)} sentences × {args.noise_copies} corruptions)")
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch,
                              collate_fn=collate_fn, shuffle=False)
        print(f"Running CTC inference on train split  beam_width={args.beam_width}...")
        train_rows = run_inference(model, train_dl, vocab, device, args.beam_width)

    write_tsv(args.out_train, train_rows)
    write_tsv(args.out_val,   val_rows)

    print("\nSample train pairs (noisy → clean):")
    for noisy, tgt in train_rows[:5]:
        print(f"  Noisy: {noisy[:80]}")
        print(f"  Gold : {tgt[:80]}")
        print()
    print("Sample val pairs (real CTC → clean):")
    for pred, tgt in val_rows[:3]:
        print(f"  CTC : {pred[:80]}")
        print(f"  Gold: {tgt[:80]}")
        print()


if __name__ == "__main__":
    main()
