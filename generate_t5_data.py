"""
Generate (ctc_prediction, target_sentence) pairs for T5 fine-tuning.

Runs the best CTC checkpoint over the training split and writes two TSV files:
  t5_train.tsv  — training pairs (noisy CTC → clean English)
  t5_val.tsv    — validation pairs for T5 evaluation

Usage:
    python generate_t5_data.py \
        --checkpoint checkpoints/model_ctc_primary.pth \
        --keypoints  /data/how2sign/ \
        --csv        /data/how2sign/train_labels.csv \
        --beam_width 10
"""
import argparse, csv, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import How2SignDataset, SignDataset
from vocab import Vocab
from model import TransformerSignModel, POSE_FEATURES
from train import collate_fn, make_padding_mask, ctc_decode, ctc_beam_search


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--keypoints",   default=None)
    p.add_argument("--csv",         default=None)
    p.add_argument("--fake",        default="dataset")
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--beam_width",  type=int,   default=10)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--out_train",   default="t5_train.tsv")
    p.add_argument("--out_val",     default="t5_val.tsv")
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
    if "norm_mean" in ckpt:
        ds_tmp = How2SignDataset(args.keypoints, args.csv) if args.keypoints else None
        if ds_tmp:
            ds_tmp.set_norm_stats(ckpt["norm_mean"].cpu(), ckpt["norm_std"].cpu())

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
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  beam_width: {args.beam_width}")

    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          collate_fn=collate_fn, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          collate_fn=collate_fn, shuffle=False)

    print("Running CTC inference on train split...")
    train_rows = run_inference(model, train_dl, vocab, device, args.beam_width)
    print("Running CTC inference on val split...")
    val_rows   = run_inference(model, val_dl,   vocab, device, args.beam_width)

    write_tsv(args.out_train, train_rows)
    write_tsv(args.out_val,   val_rows)

    # Print a few examples
    print("\nSample pairs (noisy CTC → clean target):")
    for pred, tgt in train_rows[:5]:
        print(f"  CTC : {pred[:80]}")
        print(f"  Gold: {tgt[:80]}")
        print()


if __name__ == "__main__":
    main()
