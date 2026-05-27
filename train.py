import argparse, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
                   help="Path to extracted How2Sign keypoints dir (bfh_2d_front/...)")
    p.add_argument("--csv", default=None,
                   help="How2Sign annotation CSV path")
    p.add_argument("--fake", default="dataset",
                   help="Fake local dataset dir (default: dataset/)")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch",  type=int, default=BATCH_SIZE)
    p.add_argument("--checkpoint", default="model.pth")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


def collate_fn(batch):
    poses, texts = zip(*batch)
    input_lengths = torch.tensor([p.shape[0] for p in poses], dtype=torch.long)
    padded = pad_sequence(poses, batch_first=True)
    return padded, texts, input_lengths


def ctc_decode(log_probs, vocab):
    indices = log_probs.argmax(-1).tolist()
    result, prev = [], -1
    for i in indices:
        if i != prev and i != 0:
            result.append(i)
        prev = i
    return vocab.decode(result)


def main():
    args = parse_args()
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

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = SignModel(POSE_FEATURES, 256, len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        print(f"Resumed from epoch {ckpt['epoch']}  (loss {ckpt['loss']:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        for padded, texts, input_lengths in dl:
            padded = padded.to(device)
            input_lengths = input_lengths.to(device)

            logits = model(padded)
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

            encoded = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
            targets = torch.cat(encoded)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(dl)
        scheduler.step(avg)
        print(f"Epoch {epoch:3d}  Loss: {avg:.4f}  LR: {opt.param_groups[0]['lr']:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "vocab": vocab.w2i, "loss": avg}, args.checkpoint)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_pose, sample_text, _ = next(iter(
                    DataLoader(ds, batch_size=1, collate_fn=collate_fn)
                ))
                lp = F.log_softmax(model(sample_pose.to(device)), dim=-1).squeeze(0)
                print(f"  Target: {sample_text[0]}  |  Pred: {ctc_decode(lp.cpu(), vocab)}")

    print(f"\nBest loss: {best_loss:.4f}  |  Saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
