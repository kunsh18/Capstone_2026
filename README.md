# An Approach for Decoding Commercial Intent Understanding

## Overview

This project implements a **Multimodal Intent Recognition** framework aimed at interpreting **Hindi and Hinglish** dialogues in **corporate** and **commercial** settings (e.g., grocery shops, office scenes). The design targets multiple modalities—**video (visuals)**, **audio (acoustics)**, and **text (transcripts)**—so the model can capture context and classify the intent behind utterances.

**What is in this repository:** The included training script (`multimodal_intent_training.py`) implements a **video + text** pipeline: DistilBERT-based multilingual text embeddings fused with frame-level features from **MobileNetV3-Small** over sampled video frames.

## Dataset structure

Training uses a **custom** Hindi/Hinglish dataset aligned with scene videos.

| Asset | Role |
|--------|------|
| **`Corporate.xlsx`** | Corporate-scene metadata, transcripts, labels; videos under `Corporate_Video/Neetish_CorporateScene/`. |
| **`Grocery.xlsx`** | Grocery-scene metadata; videos under `Grocery_Video/Neetish/`. |
| **`Corporate_dataset.xlsx`** | Alternate naming used in some write-ups; use the Excel file(s) your scripts expect (this code uses `Corporate.xlsx` and `Grocery.xlsx`). |
| **`Intent_class.xlsx`** | Intent taxonomy for corporate and commercial interactions (standard business talk, negotiation, casual chat, sensitive or anomalous intents, etc.). |

**Video mapping:** Each row’s **`Video ID`** maps to a file `{Video ID}.mp4` in the corresponding video directory.

**Modalities (full design):**

- **Text:** Hindi and Hinglish transcripts (script prefers `Hinglish Text`, falls back to `Hindi Text`).
- **Audio:** Raw waveforms (not wired in the current `multimodal_intent_training.py`; reserved for extensions).
- **Video:** Sequentially sampled frames (default **8** frames per clip, 224×224).

**Labels:** Read from the **`Label`** column in the Excel files; class count is inferred from unique labels in the merged valid set.

## Results

| Metric | Value |
|--------|--------|
| **Best training accuracy** | **23.04%** |
| **Validation accuracy** | **20.41%** |
| **Unseen data accuracy** (hold-out test) | **16%** |

---

## Repository contents

| File | Description |
|------|-------------|
| `multimodal_intent_training.py` | Precomputes text embeddings (DistilBERT multilingual), loads videos, trains **MultimodalIntentModel**, saves `best_multimodal_intent_model.pth` / `final_multimodal_intent_model.pth`, evaluates on a **70 / 15 / 15** train–validation–test split (seed **42**). |
| `requirements.txt` | Python dependencies for the training script. |
| `cached_data.pkl` | Created after first run: cached text embeddings and paths (regenerate by deleting this file). |
---

## Getting started

### Prerequisites

- **Python 3.8+**
- **PyTorch** and **torchvision** (CUDA optional but recommended)
- **FFmpeg** (useful for broader A/V prep; OpenCV reads `.mp4` directly for this script)
- **Hugging Face Transformers** (text encoder)
- **Pandas** + **openpyxl** (Excel I/O)

### Installation

```bash
git clone https://github.com/kunsh18/Capstone_2026.git
pip install -r requirements.txt
```

### Training

```bash
python multimodal_intent_training.py
```

On first run, the script builds **`cached_data.pkl`** (text embeddings). Rows whose video files are missing are skipped. Training uses **AdamW**, **CrossEntropyLoss**, batch size **16**, **10** epochs by default (see constants at the top of the script).

---

## References (papers)

- **MintRec framework:** [arXiv:2506.10011](https://arxiv.org/abs/2506.10011)
- **Wavelet-based multimodal representation:** [arXiv:2403.10943](https://arxiv.org/abs/2403.10943)

---

## Acknowledgments

Developed as part of capstone research on **MintRec**-style and **wavelet-based** multimodal intent recognition for Hindi/Hinglish dialogue in Indian corporate and marketplace contexts.
