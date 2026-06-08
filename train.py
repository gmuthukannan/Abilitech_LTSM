import argparse, os, time, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from dataset import SignDataset, How2SignDataset
from vocab import Vocab
from model import SignModel, TransformerSignModel, POSE_FEATURES

BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints",   default=None)
    p.add_argument("--csv",         default=None)
    p.add_argument("--fake",        default="dataset")
    p.add_argument("--model",       default="transformer",
                   choices=["bilstm", "transformer"])
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=None,
                   help="Learning rate (default: 1e-4 for transformer, 1e-3 for bilstm)")
    p.add_argument("--attn_weight", type=float, default=0.7,
                   help="Fraction of loss from attention decoder (default: 0.7)")
    p.add_argument("--checkpoint",  default="model.pth")
    p.add_argument("--resume",      default=None)
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--eval_every",  type=int,   default=5)
    p.add_argument("--norm_samples",type=int,   default=2000)
    return p.parse_args()


def collate_fn(batch):
    poses, texts = zip(*batch)
    input_lengths = torch.tensor([p.shape[0] for p in poses], dtype=torch.long)
    padded = pad_sequence(poses, batch_first=True)
    return padded, texts, input_lengths


def make_padding_mask(lengths, max_len, device):
    """Boolean mask (batch, max_len): True where positions are padding."""
    return torch.arange(max_len, device=device).unsqueeze(0) >= lengths.to(device).unsqueeze(1)


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_norm_stats(dataset, train_indices, max_samples=2000):
    indices   = list(train_indices)[:max_samples]
    all_frames = torch.cat([dataset[i][0] for i in indices], dim=0)
    mean = all_frames.mean(0)
    std  = all_frames.std(0).clamp(min=1e-6)
    return mean, std


# ── Metrics ───────────────────────────────────────────────────────────────────

def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def cer(pred, target):
    if not target:
        return 0.0 if not pred else 1.0
    return _edit_distance(list(pred), list(target)) / len(target)


def wer(pred, target):
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


# ── Decoder teacher-forcing batch builder ─────────────────────────────────────

def attn_greedy_decode(model, memory, memory_mask, vocab, max_len=300):
    """Greedy autoregressive decode using the attention decoder."""
    device = memory.device
    tgt = torch.tensor([[vocab.sos_id]], device=device)
    for _ in range(max_len):
        logits   = model.attn_dec(tgt, memory,
                                  memory_key_padding_mask=memory_mask)
        next_tok = logits[0, -1].argmax(-1).item()
        if next_tok == vocab.eos_id:
            break
        tgt = torch.cat([tgt, torch.tensor([[next_tok]], device=device)], dim=1)
    return vocab.decode(tgt[0, 1:].tolist())


def build_decoder_batch(encoded_seqs, vocab, device):
    """
    encoded_seqs: list of list[int] (character indices, no SOS/EOS)
    Returns:
        tgt_in  (batch, L)   — [SOS, c1, c2, ...]
        tgt_out (batch, L)   — [c1, c2, ..., EOS], padding=-100 (ignored by CE)
        tgt_pad_mask (batch, L) — True where padding
    """
    sos, eos = vocab.sos_id, vocab.eos_id
    ins  = [torch.tensor([sos] + e,        dtype=torch.long) for e in encoded_seqs]
    outs = [torch.tensor(e       + [eos],  dtype=torch.long) for e in encoded_seqs]
    lens = torch.tensor([len(e) + 1 for e in encoded_seqs], dtype=torch.long)
    tgt_in  = pad_sequence(ins,  batch_first=True, padding_value=0   ).to(device)
    tgt_out = pad_sequence(outs, batch_first=True, padding_value=-100).to(device)
    tgt_pad = make_padding_mask(lens, tgt_in.size(1), device)
    return tgt_in, tgt_out, tgt_pad


# ── Validation loop ───────────────────────────────────────────────────────────

def evaluate(model, loader, vocab, device):
    model.eval()
    total_ctc_cer = total_wer = total_attn_cer = 0.0
    use_attn = hasattr(model, 'attn_dec')
    with torch.no_grad():
        for padded, texts, input_lengths in loader:
            padded        = padded.to(device)
            input_lengths = input_lengths.to(device)
            if hasattr(model, 'subsampled_lengths'):
                out_lengths = model.subsampled_lengths(input_lengths)
                mask = make_padding_mask(out_lengths, (padded.size(1)+1)//2, device)
            else:
                out_lengths = input_lengths
                mask = make_padding_mask(input_lengths, padded.size(1), device)
            memory = model.encode(padded, mask) if use_attn else None
            lp     = F.log_softmax(
                model.ctc_fc(memory) if use_attn else model(padded, src_key_padding_mask=mask),
                dim=-1
            )
            for i, text in enumerate(texts):
                target = text.upper()
                ctc_pred = ctc_decode(lp[i].cpu(), vocab)
                total_ctc_cer += cer(ctc_pred, target)
                total_wer     += wer(ctc_pred, target)
                if use_attn:
                    mem_i = memory[i:i+1]
                    msk_i = mask[i:i+1] if mask is not None else None
                    attn_pred = attn_greedy_decode(model, mem_i, msk_i, vocab)
                    total_attn_cer += cer(attn_pred, target)
    n = len(loader.dataset)
    if use_attn:
        return total_ctc_cer / n, total_wer / n, total_attn_cer / n
    return total_ctc_cer / n, total_wer / n, None


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
        print(f"Fake dataset: {len(ds)} samples")

    vocab = Vocab()
    for _, t in ds:
        vocab.add(t)
    print(f"Vocab: {len(vocab)} tokens  (blank=0, sos=1, eos=2, chars=3..{len(vocab)-1})")

    val_size   = max(1, int(args.val_split * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {train_size} train / {val_size} val")

    print(f"Computing normalisation stats from up to {args.norm_samples} clips...")
    mean, std = compute_norm_stats(ds, train_ds.indices, args.norm_samples)
    ds.set_norm_stats(mean, std)
    print(f"  mean range [{mean.min():.3f}, {mean.max():.3f}]  "
          f"std range [{std.min():.3f}, {std.max():.3f}]")

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
    if args.model == "transformer":
        print(f"Attention decoder weight: {args.attn_weight}  "
              f"CTC weight: {1-args.attn_weight}")

    lr        = args.lr if args.lr is not None else (1e-4 if args.model == "transformer" else LR)
    print(f"Learning rate: {lr:.2e}")
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5, mode='min'
    )
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 0
    best_cer    = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer"  in ckpt: opt.load_state_dict(ckpt["optimizer"])
        if "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        if "norm_mean"  in ckpt: ds.set_norm_stats(ckpt["norm_mean"], ckpt["norm_std"])
        start_epoch = ckpt["epoch"] + 1
        best_cer    = ckpt.get("cer", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  "
              f"LR: {opt.param_groups[0]['lr']:.2e}  CER: {best_cer:.4f}")

    train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        num_valid  = 0
        skipped    = 0

        for padded, texts, input_lengths in train_dl:
            padded        = padded.to(device)
            input_lengths = input_lengths.to(device)

            if not torch.isfinite(padded).all():
                skipped += 1
                continue

            # Compute encoder output lengths and padding mask
            if hasattr(model, 'subsampled_lengths'):
                ctc_lengths = model.subsampled_lengths(input_lengths)
                sub_T = (padded.size(1) + 1) // 2
                mask  = make_padding_mask(ctc_lengths, sub_T, device)
            else:
                ctc_lengths = input_lengths
                mask = make_padding_mask(input_lengths, padded.size(1), device)

            # Encode targets
            encoded        = [vocab.encode(t) for t in texts]
            target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
            targets        = torch.cat([torch.tensor(e, dtype=torch.long) for e in encoded])

            # ── Forward pass ──────────────────────────────────────────────
            if hasattr(model, 'forward_train'):
                # Transformer: hybrid CTC + Attention
                tgt_in, tgt_out, tgt_pad = build_decoder_batch(encoded, vocab, device)
                ctc_logits, attn_logits  = model.forward_train(
                    padded, tgt_in,
                    src_key_padding_mask=mask,
                    tgt_key_padding_mask=tgt_pad,
                )
                log_probs    = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
                ctc_loss_val = ctc_loss_fn(log_probs, targets, ctc_lengths, target_lengths)
                attn_loss_val = F.cross_entropy(
                    attn_logits.reshape(-1, len(vocab)),
                    tgt_out.reshape(-1),
                    ignore_index=-100,
                )
                # Normalise CTC by avg target length so both losses are on same scale
                ctc_norm = ctc_loss_val / target_lengths.float().mean().clamp(min=1)
                if torch.isfinite(ctc_norm) and torch.isfinite(attn_loss_val):
                    loss = (1 - args.attn_weight) * ctc_norm + args.attn_weight * attn_loss_val
                elif torch.isfinite(attn_loss_val):
                    loss = attn_loss_val
                else:
                    loss = ctc_norm
            else:
                # BiLSTM: CTC only
                logits    = model(padded, src_key_padding_mask=mask)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss      = ctc_loss_fn(log_probs, targets, ctc_lengths, target_lengths)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            opt.zero_grad()
            loss.backward()
            clip = 1.0 if args.model == "transformer" else 5.0
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not torch.isfinite(grad_norm):
                opt.zero_grad()
                skipped += 1
                continue

            opt.step()
            total_loss += loss.item()
            num_valid  += 1

        avg      = total_loss / max(num_valid, 1)
        skip_str = f"  skipped: {skipped}/{len(train_dl)}" if skipped else ""
        print(f"Epoch {epoch:3d}  Loss: {avg:.4f}  LR: {opt.param_groups[0]['lr']:.2e}{skip_str}", end="")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_cer, val_wer, val_attn_cer = evaluate(model, val_dl, vocab, device)
            best_eval_cer = min(val_cer, val_attn_cer) if val_attn_cer is not None else val_cer
            scheduler.step(best_eval_cer)
            attn_str = f"  ATT-CER: {val_attn_cer:.4f}" if val_attn_cer is not None else ""
            print(f"  CTC-CER: {val_cer:.4f}  WER: {val_wer:.4f}{attn_str}", end="")

            if best_eval_cer < best_cer:
                best_cer = best_eval_cer
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
