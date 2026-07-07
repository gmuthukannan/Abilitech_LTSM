"""
Evaluate a saved checkpoint on the How2Sign validation split.
Optionally applies a fine-tuned T5 corrector on top of CTC output.

Usage:
    # CTC only (beam search)
    python eval.py \
        --checkpoint checkpoints/model_ctc_primary.pth \
        --keypoints  /data/how2sign/ \
        --csv        /data/how2sign/train_labels.csv \
        --beam_width 10

    # CTC + T5 correction (full pipeline)
    python eval.py \
        --checkpoint checkpoints/model_ctc_primary.pth \
        --t5         checkpoints/t5_corrector/ \
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

PREFIX = "fix asl: "


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
    p.add_argument("--t5",          default=None,
                   help="Path to fine-tuned T5 corrector directory (optional)")
    p.add_argument("--t5_batch",    type=int,   default=32)
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

    # ── CTC Evaluate ─────────────────────────────────────────────────────────
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
    print(f"CTC eval time: {elapsed:.1f}s")

    # ── T5 Correction (optional) ──────────────────────────────────────────────
    if args.t5:
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        print(f"\nLoading T5 corrector from {args.t5} ...")
        t5_tok   = T5TokenizerFast.from_pretrained(args.t5)
        t5_model = T5ForConditionalGeneration.from_pretrained(args.t5).to(device)
        t5_model.eval()

        decode_fn = (lambda lp: ctc_beam_search(lp, vocab, args.beam_width)) \
                    if args.beam_width > 1 else (lambda lp: ctc_decode(lp, vocab))

        all_ctc_preds, all_targets = [], []
        model.eval()
        with torch.no_grad():
            for padded, texts, input_lengths in val_dl:
                padded        = padded.to(device)
                input_lengths = input_lengths.to(device)
                if hasattr(model, 'subsampled_lengths'):
                    out_lengths = model.subsampled_lengths(input_lengths)
                    sub_T = model.subsampled_lengths(
                        input_lengths.new_tensor([padded.size(1)])).item()
                    mask = make_padding_mask(out_lengths, sub_T, device)
                else:
                    mask = make_padding_mask(input_lengths, padded.size(1), device)
                memory = model.encode(padded, mask)
                lp = F.log_softmax(model.ctc_fc(memory), dim=-1)
                for i, text in enumerate(texts):
                    all_ctc_preds.append(decode_fn(lp[i].cpu()))
                    all_targets.append(text.upper())

        # Run T5 in mini-batches
        t0 = time.time()
        total_t5_cer = 0.0
        for start in range(0, len(all_ctc_preds), args.t5_batch):
            batch_preds = all_ctc_preds[start:start + args.t5_batch]
            batch_tgts  = all_targets[start:start + args.t5_batch]
            enc = t5_tok(
                [PREFIX + p for p in batch_preds],
                max_length=128, truncation=True, padding=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = t5_model.generate(
                    **enc, max_new_tokens=128, num_beams=4
                )
            corrected = t5_tok.batch_decode(out, skip_special_tokens=True)
            for c, t in zip(corrected, batch_tgts):
                total_t5_cer += cer(c.upper(), t)

        t5_cer = total_t5_cer / max(len(all_ctc_preds), 1)
        print(f"T5-CER   (pipeline):  {t5_cer:.4f}   "
              f"(Δ {ctc_cer - t5_cer:+.4f} vs CTC beam)")
        print(f"T5 eval time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
