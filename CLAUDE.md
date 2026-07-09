# ASL Sign-to-Text Pipeline

Transcribes American Sign Language (ASL) video clips into natural English using a staged deep learning pipeline.

---

## Target Architecture

```
Video Input
    └─► MediaPipe Keypoints (225-dim per frame)
            └─► Transformer Encoder + CTC Loss
                    └─► Beam Search Decode
                            └─► [Optional] T5 Correction
                                    └─► Natural English Output
                                            └─► [Optional] Claude API Refinement
```

---

## Roadmap

### Phase 1 — LSTM Baseline ✅ Done

- [x] BiLSTM model with CTC loss (`model.py`, `train.py`)
- [x] How2Sign OpenPose JSON dataset loader (`How2SignDataset`)
- [x] Checkpoint save/resume
- [x] Switched from word-level (~26.9k tokens) to character-level vocabulary for stable CTC training
- [x] Verified inference pipeline and greedy CTC decode

### Phase 2 — Transformer Encoder ✅ Done — Best CER: 0.7497

- [x] Replaced BiLSTM with Transformer encoder + hybrid CTC/Attention loss
- [x] Stride-1 Conv1D subsampler (tripled usable training data: 8.8k → 30k clips)
- [x] CTC as primary loss (attn_weight=0.3) — biggest single improvement (0.85 → 0.75 CER)
- [x] CosineAnnealingLR, AdamW, label smoothing, temporal masking augmentation
- [x] CTC beam search decoder (`ctc_beam_search` in `train.py`)
- [x] Standalone evaluation script (`eval.py`) with `--beam_width` and `--t5` flags

**Best checkpoint:** `model_ctc_primary.pth` — CER 0.7497 (greedy), ~0.749 (beam=10)

**Key finding:** CTC as primary signal (70% CTC + 30% Attention) was the critical unlock.
Training with 70% attention loss but evaluating on CTC caused the ~0.85 plateau.

### Phase 3 — T5 Correction ✅ Done (T5 not yet beating CTC)

- [x] `generate_t5_data.py` — generates (noisy_input, clean_target) pairs
  - Default mode: runs CTC inference on training clips
  - `--synthetic` mode: corrupts clean sentences at ~0.75 CER to fix train/val mismatch
- [x] `train_t5.py` — fine-tunes flan-t5-base on correction pairs
- [x] End-to-end eval in `eval.py` with `--t5 <dir>` flag

**Results:**

| Model | CER |
|-------|-----|
| CTC greedy | 0.7497 |
| CTC beam=10 | ~0.749 |
| T5-small (real CTC data) | 0.8282 — worse |
| T5-small (synthetic noise) | 0.7760 — worse |
| T5-base (synthetic noise) | **0.7597** — still slightly worse |

**Finding:** T5 doesn't improve CER at 0.75 noise level — 75% of characters wrong is too
noisy for reconstruction. T5 becomes useful when CTC CER drops below ~0.50.
Current production pipeline: CTC + beam search only (no T5).

**Best T5 checkpoint:** `gs://abilitechhow2sign/checkpoints/t5_corrector_v4/`

### Phase 4 — Claude API Refinement ⏳ Deferred

Post-process output with Claude API. Deferred until CTC CER improves further.

### Phase 5 — Live Camera Inference 🔄 Next

Real-time webcam inference using MediaPipe Holistic for keypoint extraction,
feeding directly into the trained model with no video file I/O.

---

## Current Architecture (Phase 2)

```
How2Sign OpenPose JSON → 201-dim features
    → Conv1DSubsampler (stride=1, kernel=3)
    → PositionalEncoding
    → TransformerEncoder (4 layers, d=256, nhead=4, ff=1024)
    → CTC head → Beam Search → Text
    → [+ Attention Decoder for training regularisation]
```

### Model (`model.py`)

| Component | Detail |
|-----------|--------|
| Input | 201-dim OpenPose feature vector per frame |
| Subsampler | Conv1D stride=1 (no temporal reduction) |
| Encoder | 4-layer Transformer, d_model=256, nhead=4, ff=1024 |
| CTC head | Linear(256 → vocab_size) |
| Attn decoder | 2-layer Transformer decoder (training only) |
| Parameters | ~5.5M |

**Feature breakdown (201 dims):**
- 25 body keypoints × 3 (x, y, confidence) = 75
- 21 left-hand keypoints × 3 = 63
- 21 right-hand keypoints × 3 = 63

### Vocabulary (`vocab.py`)

Character-level tokenization. Index 0 = CTC blank, 1 = SOS, 2 = EOS, 3+ = characters.
All text uppercased. Vocab size ~70 tokens.

---

## GCS Checkpoints

```
gs://abilitechhow2sign/checkpoints/
  model_ctc_primary.pth    ← BEST — CER 0.7497, use this for inference
  model_stride1.pth        ← CER 0.8519 (older baseline)
  model_hybrid.pth         ← earlier baseline
  t5_corrector_v4/         ← flan-t5-base, val CER 0.7597

gs://abilitechhow2sign/t5_data/
  t5_train.tsv             ← ~90k synthetic noise training pairs
  t5_val.tsv               ← real CTC val predictions
```

---

## Data

### Training Data — GCloud Bucket

```
gs://abilitechhow2sign/
```

How2Sign dataset stored in Google Cloud Storage. Pull with `gsutil` before training.

### How2Sign OpenPose Keypoints (`How2SignDataset`)

Expected layout after extracting `train_2D_keypoints.tar.gz`:
```
<keypoints_dir>/
  openpose_output/
    json/
      <SENTENCE_NAME>/
        *.json        ← one file per frame
```

CSV format: tab-separated, requires `SENTENCE_NAME` and `SENTENCE` columns.
The keypoints extract directly into `/data/how2sign/` (not a subdirectory).
Use `--keypoints /data/how2sign/` — `How2SignDataset` appends `openpose_output/json/` internally.

### Fake / Local Dataset (`dataset/`)

8 synthetic samples for local CPU testing, generated by `make_data.py`.

---

## Training (`train.py`)

### Hyperparameters (Transformer, current best)

| Parameter | Value |
|-----------|-------|
| Model | Transformer encoder |
| d_model | 256 |
| Layers | 4 |
| Heads | 4 |
| Batch size | 64 (H100) |
| Epochs | 500 |
| Learning rate | 1e-4 |
| LR scheduler | CosineAnnealingLR |
| Optimizer | AdamW, weight_decay=0.01 |
| attn_weight | 0.3 (CTC gets 70% of gradient) |
| Label smoothing | 0.1 |
| Temporal masking | 2 segments, max 20 frames |
| Feature noise | std=0.01 |
| Gradient clipping | 1.0 |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--keypoints` | None | Path to How2Sign keypoints dir |
| `--csv` | None | How2Sign annotation CSV |
| `--model` | `transformer` | `transformer` or `bilstm` |
| `--epochs` | 50 | Training epochs |
| `--batch` | 16 | Batch size |
| `--checkpoint` | `model.pth` | Output path |
| `--resume` | None | Resume from checkpoint |
| `--val_split` | 0.1 | Validation fraction |
| `--eval_every` | 5 | Eval interval (epochs) |
| `--attn_weight` | 0.3 | Attention decoder loss weight |
| `--scheduler` | `cosine` | `cosine` or `plateau` |
| `--beam_width` | 1 | CTC beam width for eval (1=greedy) |
| `--weight_decay` | 0.01 | AdamW weight decay |
| `--feat_noise` | 0.01 | Gaussian noise std on features |
| `--time_mask_n` | 2 | Number of temporal mask segments |
| `--time_mask_t` | 20 | Max frames per mask segment |

### Metrics

- **CER** — Character Error Rate (primary). Best checkpoint saved on lowest val CER.
- **WER** — Word Error Rate.

---

## Evaluation (`eval.py`)

```bash
# CTC only (beam search)
python eval.py \
  --checkpoint checkpoints/model_ctc_primary.pth \
  --keypoints  /data/how2sign/ \
  --csv        /data/how2sign/train_labels.csv \
  --beam_width 10

# Full pipeline (CTC + T5)
python eval.py \
  --checkpoint checkpoints/model_ctc_primary.pth \
  --t5         checkpoints/t5_corrector_v4/ \
  --keypoints  /data/how2sign/ \
  --csv        /data/how2sign/train_labels.csv \
  --beam_width 1
```

---

## T5 Correction (`train_t5.py`, `generate_t5_data.py`)

```bash
# Generate training data (synthetic noise mode — recommended)
python generate_t5_data.py \
  --checkpoint checkpoints/model_ctc_primary.pth \
  --keypoints  /data/how2sign/ \
  --csv        /data/how2sign/train_labels.csv \
  --synthetic

# Train T5
python train_t5.py \
  --train t5_train.tsv \
  --val   t5_val.tsv \
  --epochs 20 \
  --out   checkpoints/t5_corrector/

# Back up T5 model
gsutil cp -r checkpoints/t5_corrector/ gs://abilitechhow2sign/checkpoints/t5_corrector/
```

---

## Setup

```bash
# CPU — local development
pip install torch torchvision torchaudio
pip install -r requirements.txt

# GPU — H100 (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Python version:** 3.12

---

## Running on NVIDIA Brev (H100)

### 1. Launch an instance (local machine)

```bash
brev create asl-train --instance-type g6e.12xlarge
brev ls   # wait until status = running
```

### 2. Open a shell (local machine)

```bash
brev shell abilitech-test-007
# or
brev open asl-train   # opens VS Code remote
```

### 3. Authenticate GCloud (one-time per instance)

```bash
gcloud auth login
gcloud config set project <your-project-id>
```

If `gcloud` is not installed:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### 4. Pull training data from GCloud

```bash
sudo mkdir -p /data/how2sign
sudo chmod 777 /data/how2sign

gsutil cp gs://abilitechhow2sign/train_2D_keypoints.tar.gz /data/how2sign/train_2D_keypoints.tar.gz
tar -xzf /data/how2sign/train_2D_keypoints.tar.gz -C /data/how2sign/
gsutil cp gs://abilitechhow2sign/train_labels.csv /data/how2sign/
```

### 5. Set up repo and Python environment

```bash
git clone https://github.com/gmuthukannan/Abilitech_LTSM ~/Abilitech_LTSM
cd ~/Abilitech_LTSM
# or if already cloned:
cd ~/Abilitech_LTSM && git pull
```

```bash
sudo apt install python3.12-venv -y
python3 -m venv ~/asl/venv
source ~/asl/venv/bin/activate
echo "source ~/asl/venv/bin/activate" >> ~/.bashrc

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 6. Pull checkpoints from GCloud

```bash
mkdir -p ~/Abilitech_LTSM/checkpoints
gsutil cp gs://abilitechhow2sign/checkpoints/model_ctc_primary.pth ~/Abilitech_LTSM/checkpoints/model_ctc_primary.pth
# Optional: T5 corrector
gsutil cp -r gs://abilitechhow2sign/checkpoints/t5_corrector_v4/ ~/Abilitech_LTSM/checkpoints/t5_corrector_v4/
```

### 7. Resume CTC training

```bash
tmux new -s train

python train.py \
  --keypoints  /data/how2sign/ \
  --csv        /data/how2sign/train_labels.csv \
  --model      transformer \
  --batch      64 \
  --epochs     500 \
  --checkpoint checkpoints/model_ctc_primary_v2.pth \
  --resume     checkpoints/model_ctc_primary.pth

# Detach: Ctrl+B then D
# Reattach: tmux attach -t train
```

### 8. Push checkpoints back to GCloud

```bash
gsutil cp ~/Abilitech_LTSM/checkpoints/model_ctc_primary.pth gs://abilitechhow2sign/checkpoints/model_ctc_primary.pth
gsutil cp -r ~/Abilitech_LTSM/checkpoints/t5_corrector_v4/ gs://abilitechhow2sign/checkpoints/t5_corrector_v4/
```

### 9. Stop the instance when done (local machine)

```bash
brev stop asl-train
brev delete asl-train   # only when fully done to avoid charges
```

---

## File Map

| File | Purpose |
|------|---------|
| `model.py` | BiLSTM (Phase 1) + TransformerSignModel (Phase 2) |
| `dataset.py` | `SignDataset` (local) + `How2SignDataset` (OpenPose JSON) |
| `vocab.py` | Character-level vocabulary with CTC blank/SOS/EOS |
| `train.py` | Training loop, CTC+Attention loss, beam search, checkpointing |
| `eval.py` | Standalone evaluation: CTC beam search + optional T5 pipeline |
| `generate_t5_data.py` | Generate (noisy CTC, clean target) pairs for T5 fine-tuning |
| `train_t5.py` | Fine-tune flan-t5-base on correction pairs |
| `extract_keypoints.py` | MediaPipe Holistic extraction from raw `.mp4` video |
| `make_data.py` | Generate synthetic local dataset for testing |
| `requirements.txt` | Python dependencies |
| `dataset/` | 8 fake samples (pose.npy + text.txt each) |
