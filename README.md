# ASL Sign-to-Text — Brev Training Setup

Step-by-step guide to spin up a fresh Brev H100 instance and run training.

---

## 1. Launch a Brev Instance (local machine)

```bash
brev create asl-train --instance-type g6e.12xlarge
brev ls   # wait until status is RUNNING
```

---

## 2. Open a Shell

```bash
brev shell asl-train
```

---

## 3. Install gcloud SDK (if not present)

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

Skip this step if `gcloud --version` already works.

---

## 4. Authenticate with GCP

```bash
gcloud auth login
gcloud config set project <your-project-id>
```

---

## 5. Download Training Data from GCS Bucket

```bash
mkdir -p /data/how2sign

# OpenPose keypoints (~several GB)
gsutil -m cp -r gs://abilitechhow2sign/train_2D_keypoints.tar.gz /data/how2sign/

# Annotation CSV
gsutil cp gs://abilitechhow2sign/train_labels.csv /data/how2sign/
```

---

## 6. Extract Keypoints

```bash
tar -xzf /data/how2sign/train_2D_keypoints.tar.gz -C /data/how2sign/
```

**Expected layout after extraction:**
```
/data/how2sign/
  openpose_output/
    json/
      --7E2sU6zP4_10-5-rgb_front/
        *.json
      --7E2sU6zP4_11-5-rgb_front/
        ...
  train_labels.csv
  train_2D_keypoints.tar.gz
```

**Verify:**
```bash
ls /data/how2sign/openpose_output/json/ | head -5
head -3 /data/how2sign/train_labels.csv
```

The folder names under `json/` should match the `SENTENCE_NAME` column in the CSV.

---

## 7. Clone Repo and Install Dependencies

```bash
git clone https://github.com/gmuthukannan/Abilitech_LTSM ~/asl
cd ~/asl
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 8. Run Training

```bash
mkdir -p checkpoints

# Phase 2 — Transformer (default)
python3 train.py \
  --keypoints /data/how2sign/ \
  --csv       /data/how2sign/train_labels.csv \
  --model     transformer \
  --epochs    200 \
  --batch     64 \
  --checkpoint checkpoints/model_transformer.pth
```

**Check the output line:** `Model: transformer  |  Parameters: ~6M` confirms Phase 2 is running.

### Resume after interruption

```bash
python3 train.py \
  --keypoints /data/how2sign/ \
  --csv       /data/how2sign/train_labels.csv \
  --model     transformer \
  --epochs    200 \
  --batch     64 \
  --checkpoint checkpoints/model_transformer.pth \
  --resume    checkpoints/model_transformer.pth
```

> **Note:** Only resume from a checkpoint trained with character-level vocab. Do not resume from an old word-level checkpoint — the model output dimension will mismatch (~26,940 vs ~30).

---

## 9. Save Checkpoint Before Deleting Instance

```bash
# Option A — copy to local machine
brev scp asl-train:~/asl/checkpoints/model_char.pth ./checkpoints/

# Option B — push to GCS
gsutil cp ~/asl/checkpoints/model_char.pth gs://abilitechhow2sign/checkpoints/
```

---

## 10. Re-activating the venv on Reconnect

Every time you SSH back into the instance:

```bash
cd ~/asl && source venv/bin/activate
```

---

## 11. Stop / Delete Instance

```bash
brev stop asl-train
brev delete asl-train   # avoids ongoing charges
```

---

## Key Paths Reference

| What | Path on Brev |
|------|-------------|
| Keypoints root | `/data/how2sign/` |
| JSON clips | `/data/how2sign/openpose_output/json/` |
| Annotation CSV | `/data/how2sign/train_labels.csv` |
| Checkpoints | `~/asl/checkpoints/` |
| GCS bucket | `gs://abilitechhow2sign/` |
