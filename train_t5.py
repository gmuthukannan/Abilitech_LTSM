"""
Fine-tune T5-small to correct noisy CTC output into fluent English.

Input  (T5 encoder): "fix asl: I LVE IN NEW YRK"
Output (T5 decoder): "I live in New York"

Usage:
    python train_t5.py \
        --train  t5_train.tsv \
        --val    t5_val.tsv \
        --epochs 20 \
        --out    checkpoints/t5_corrector/
"""
import argparse, os, time, csv, torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import get_cosine_schedule_with_warmup


PREFIX = "fix asl: "   # task prefix that tells T5 what to do
MODEL_NAME = "google/flan-t5-base"    # 250M params — more capacity for 0.75 CER reconstruction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",    required=True,  help="t5_train.tsv from generate_t5_data.py")
    p.add_argument("--val",      required=True,  help="t5_val.tsv")
    p.add_argument("--out",      default="checkpoints/t5_corrector")
    p.add_argument("--model",    default=MODEL_NAME)
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--max_src",  type=int,   default=128,  help="Max input token length")
    p.add_argument("--max_tgt",  type=int,   default=128,  help="Max target token length")
    return p.parse_args()


class CorrectionDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_src, max_tgt):
        self.pairs = []
        with open(tsv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                inp = row["input"].strip()
                tgt = row["target"].strip()
                if inp and tgt:
                    self.pairs.append((inp, tgt))
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, tgt = self.pairs[idx]
        src_enc = self.tokenizer(
            PREFIX + inp,
            max_length=self.max_src, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt,
            max_length=self.max_tgt, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        labels = tgt_enc["input_ids"].squeeze()
        # T5 convention: replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      src_enc["input_ids"].squeeze(),
            "attention_mask": src_enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


def compute_cer(pred, target):
    def edit(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]
    return edit(list(pred), list(target)) / max(len(target), 1)


def evaluate(model, loader, tokenizer, device, max_tgt):
    model.eval()
    total_cer, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tgt,
                num_beams=4,
            )
            preds   = tokenizer.batch_decode(generated,        skip_special_tokens=True)
            targets = tokenizer.batch_decode(
                labels.masked_fill(labels == -100, tokenizer.pad_token_id),
                skip_special_tokens=True
            )
            for p, t in zip(preds, targets):
                total_cer += compute_cer(p.upper(), t.upper())
                n += 1
    return total_cer / max(n, 1)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}")

    tokenizer = T5TokenizerFast.from_pretrained(args.model)
    model     = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"T5 parameters: {n_params:,}")

    train_ds = CorrectionDataset(args.train, tokenizer, args.max_src, args.max_tgt)
    val_ds   = CorrectionDataset(args.val,   tokenizer, args.max_src, args.max_tgt)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps   = len(train_dl) * args.epochs
    warmup_steps  = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    os.makedirs(args.out, exist_ok=True)
    best_cer = float("inf")
    t_start  = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            loss = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_cer  = evaluate(model, val_dl, tokenizer, device, args.max_tgt)
        lr_now   = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}  Loss: {avg_loss:.4f}  LR: {lr_now:.2e}  Val-CER: {val_cer:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            model.tie_weights()   # re-sync lm_head → shared.weight before saving
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"  ✓ Saved to {args.out}  (best CER: {best_cer:.4f})")

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\nBest T5 val CER: {best_cer:.4f}")
    print(f"Training time: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
