import argparse, os, time, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from dataset import SignDataset, How2SignDataset
from vocab import Vocab
from model import SignModel, TransformerSignModel, POSE_FEATURES

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints", default=None,
                   help="Path to extracted How2Sign keypoints dir")
    p.add_argument("--csv", default=None,
                   help="How2Sign annotation CSV path")
    p.add_argument("--fake", default="dataset",
                   help="Fake local dataset dir (default: dataset/)")
    p.add_argument("--model", default="transformer",
                   choices=["bilstm", "transformer"],
                   help="Model architecture (default: transformer)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch",  type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 1e-4 for transformer, 1e-3 for bilstm)")
    p.add_argument("--checkpoint", default="model.pth")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of data held out for validation (default: 0.1)")
    p.add_argument("--eval_every", type=int, default=5,
                   help="Run CER/WER eval every N epochs (default: 5)")
    p.add_argument("--norm_samples", type=int, default=2000,
                   help="Number of training clips to sample for normalisation stats")
    return p.parse_args()


def collate_fn(batch):
    poses, texts = zip(*batch)
    input_lengths = torch.tensor([p.shape[0] for p in poses], dtype=torch.long)
    padded = pad_sequence(poses, batch_first=True)
    return padded, texts, input_lengths


def make_padding_mask(input_lengths, max_len, device):
    """Boolean mask (batch, max_len): True where frames are padding."""
    return torch.arange(max_len, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1).to(device)


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_norm_stats(dataset, train_indices, max_samples=2000):
    """Compute per-feature mean and std from a sample of training clips."""
    indices = list(train_indices)[:max_samples]
    all_frames = []
    for i in indices:
        pose, _ = dataset[i]
        all_frames.append(pose)
    all_frames = torch.cat(all_frames, dim=0)  # (total_frames, features)
    mean = all_frames.mean(0)
    std  = all_frames.std(0).clamp(min=1e-6)
    return mean, std


# ── Metrics ──────────────────────────────────────────────────────────────────

def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return _edit_distance(list(pred), list(target)) / len(target)


def wer(pred: str, target: str) -> float:
    p_words, t_words = pred.split(), target.split()
    if not t_words:
        return 0.0 if not p_words else 1.0
    return _edit_distance(p_words, t_words) / len(t_words)


# ── Greedy CTC decode ─────────────────────────────────────────────────────────

def ctc_decode(log_probs, vocab):
    indices = log_probs.argmax(-1).tolist()
    result, prev = [], -1
    for i in indices:
        if i != prev and i != 0:
            result.append(i)
        prev = i
    return vocab.decode(result)


# ── Validation loop ───────────────────────────────────────────────────────────

def evaluate(model, loader, vocab, device):
    model.eval()
    total_cer = total_wer = 0.0
    with torch.no_grad():
        for padded, texts, input_lengths in loader:
            padded = padded.to(device)
            mask   = make_padding_mask(input_lengths, padded.size(1), device)
            lp     = F.log_softmax(model(padded, src_key_padding_mask=mask), dim=-1)
            for i, text in enumerate(texts):
                pred   = ctc_decode(lp[i].cpu(), vocab)
                target = text.upper()
                total_cer += cer(pred, target)
                total_wer += wer(pred, target)
    n = len(loader.dataset)
    return total_cer / n, total_wer / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.keypoints and args.csv:
        ds = How2SignDataset(args.keypoints, args.csv)
        print(f"How2Sign dataset: {len(ds)} clips")
    else:
        ds = SignDataset(args.fake)
        print(f"Fake dataset: {len(ds)} samples (use --keypoints + --csv for real data)")

    vocab = Vocab()
    for _, t in ds:
        vocab.add(t)
    print(f"Vocab: {len(vocab)} tokens")

    val_size   = max(1, int(args.val_split * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {train_size} train / {val_size} val")

    print(f"Computing normalisation stats from up to {args.norm_samples} training clips...")
    mean, std = compute_norm_stats(ds, train_ds.indices, max_samples=args.norm_samples)
    ds.set_norm_stats(mean, std)
    print(f"  Feature mean range: [{mean.min():.3f}, {mean.max():.3f}]  "
          f"std range: [{std.min():.3f}, {std.max():.3f}]")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

    if args.model == "transformer":
        model = TransformerSignModel(POSE_FEATURES, d_model=256, nhead=4,
                                     num_layers=4, dim_feedforward=1024,
                                     dropout=0.1, out_size=len(vocab)).to(device)
    else:
        model = SignModel(POSE_FEATURES, hidden=256, out_size=len(vocab)).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}")

    lr = args.lr if args.lr is not None else (1e-4 if args.model == "transformer" else LR)
    print(f"Learning rate: {lr:.2e}")
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    ctc_loss  = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 0
    best_cer    = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_cer    = ckpt.get("cer", float("inf"))
        if "norm_mean" in ckpt and "norm_std" in ckpt:
            ds.set_norm_stats(ckpt["norm_mean"], ckpt["norm_std"])
            print("Restored norm stats from checkpoint")
        print(f"Resumed from epoch {ckpt['epoch']}  "
              f"LR: {opt.param_groups[0]['lr']:.2e}  CER: {best_cer:.4f}")

    train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        skipped = 0
        for padded, texts, input_lengths in train_dl:
            padded        = padded.to(device)
            input_lengths = input_lengths.to(device)

            # Guard 1: skip batches with NaN/inf input features
            if not torch.isfinite(padded).all():
                skipped += 1
                continue

            mask      = make_padding_mask(input_lengths, padded.size(1), device)
            logits    = model(padded, src_key_padding_mask=mask)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            encoded        = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
            targets        = torch.cat(encoded)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Guard 2: skip batches with NaN/inf loss
            if not torch.isfinite(loss):
                skipped += 1
                continue

            opt.zero_grad()
            loss.backward()
            clip = 1.0 if args.model == "transformer" else 5.0

            # Guard 3: skip if gradients are NaN/inf (prevents weight corruption)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not torch.isfinite(grad_norm):
                opt.zero_grad()
                skipped += 1
                continue

            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(train_dl)
        scheduler.step(avg)
        skip_str = f"  skipped: {skipped}/{len(train_dl)}" if skipped else ""
        print(f"Epoch {epoch:3d}  Loss: {avg:.4f}  LR: {opt.param_groups[0]['lr']:.2e}{skip_str}", end="")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_cer, val_wer = evaluate(model, val_dl, vocab, device)
            print(f"  CER: {val_cer:.4f}  WER: {val_wer:.4f}", end="")

            if val_cer < best_cer:
                best_cer = val_cer
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "model_type": args.model,
                    "vocab":      vocab.w2i,
                    "loss":       avg,
                    "cer":        val_cer,
                    "wer":        val_wer,
                    "norm_mean":  mean.cpu(),
                    "norm_std":   std.cpu(),
                    "optimizer":  opt.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                }, args.checkpoint)
                print("  [saved]", end="")

        print()

    elapsed = time.time() - train_start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\nBest val CER: {best_cer:.4f}  |  Saved to {args.checkpoint}")
    print(f"Training time: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
