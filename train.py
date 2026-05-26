import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset import SignDataset
from vocab import Vocab
from model import SignModel, POSE_FEATURES

BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3


def collate_fn(batch):
    poses, texts = zip(*batch)
    input_lengths = torch.tensor([p.shape[0] for p in poses], dtype=torch.long)
    padded_poses = pad_sequence(poses, batch_first=True)  # (N, T_max, F)
    return padded_poses, texts, input_lengths


def ctc_decode(log_probs, vocab):
    """Greedy CTC decode: collapse repeated tokens and remove blanks."""
    indices = log_probs.argmax(-1).tolist()
    result, prev = [], -1
    for i in indices:
        if i != prev and i != 0:
            result.append(i)
        prev = i
    return vocab.decode(result)


ds = SignDataset("dataset")
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

vocab = Vocab()
for _, t in ds:
    vocab.add(t)

print(f"Dataset: {len(ds)} samples  |  Vocab: {len(vocab)} tokens")

model = SignModel(POSE_FEATURES, 256, len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=LR)
ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for padded_poses, texts, input_lengths in dl:
        logits = model(padded_poses)                      # (N, T, V)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)           # (T, N, V) for CTC

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
    print(f"Epoch {epoch:2d}  Loss: {avg:.4f}")

    # Sample prediction every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            pose_sample, text_sample, length_sample = next(iter(
                DataLoader(ds, batch_size=1, collate_fn=collate_fn)
            ))
            logits = model(pose_sample)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            pred = ctc_decode(log_probs, vocab)
            print(f"  Target: {text_sample[0]}  |  Pred: {pred}")
