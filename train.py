import argparse, os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from dataset import SignDataset, How2SignDataset
from vocab import Vocab
from model import SignModel, POSE_FEATURES

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
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch",  type=int, default=BATCH_SIZE)
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
    """Levenshtein distance between two sequences (strings or word lists)."""
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
    """Character Error Rate — edit distance at character level, normalised by target length."""
    if not target:
        return 0.0 if not pred else 1.0
    return _edit_distance(list(pred), list(target)) / len(target)


def wer(pred: str, target: str) -> float:
    """Word Error Rate — edit distance at word level, normalised by target word count."""
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
        for padded, texts, _ in loader:
            lp = F.log_softmax(model(padded.to(device)), dim=-1)
            for i, text in enumerate(texts):
                pred = ctc_decode(lp[i].cpu(), vocab)
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

    # Compute normalisation stats from training clips and apply to dataset
    print(f"Computing normalisation stats from up to {args.norm_samples} training clips...")
    mean, std = compute_norm_stats(ds, train_ds.indices, max_samples=args.norm_samples)
    ds.set_norm_stats(mean, std)
    print(f"  Feature mean range: [{mean.min():.3f}, {mean.max():.3f}]  "
          f"std range: [{std.min():.3f}, {std.max():.3f}]")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model    = SignModel(POSE_FEATURES, 256, len(vocab)).to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    ctc_loss  = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 0
    best_cer    = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_cer    = ckpt.get("cer", float("inf"))
        # Restore normalisation stats from checkpoint if present
        if "norm_mean" in ckpt and "norm_std" in ckpt:
            ds.set_norm_stats(ckpt["norm_mean"], ckpt["norm_std"])
            print(f"Restored norm stats from checkpoint")
        print(f"Resumed from epoch {ckpt['epoch']}  (CER {best_cer:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        for padded, texts, input_lengths in train_dl:
            padded        = padded.to(device)
            input_lengths = input_lengths.to(device)

            logits    = model(padded)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            encoded        = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
            targets        = torch.cat(encoded)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(train_dl)
        scheduler.step(avg)
        print(f"Epoch {epoch:3d}  Loss: {avg:.4f}  LR: {opt.param_groups[0]['lr']:.2e}", end="")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_cer, val_wer = evaluate(model, val_dl, vocab, device)
            print(f"  CER: {val_cer:.4f}  WER: {val_wer:.4f}", end="")

            if val_cer < best_cer:
                best_cer = val_cer
                torch.save({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "vocab":     vocab.w2i,
                    "loss":      avg,
                    "cer":       val_cer,
                    "wer":       val_wer,
                    "norm_mean": mean.cpu(),
                    "norm_std":  std.cpu(),
                }, args.checkpoint)
                print("  [saved]", end="")

        print()

    print(f"\nBest val CER: {best_cer:.4f}  |  Saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
