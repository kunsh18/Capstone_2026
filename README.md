# Multimodal Intent Recognition for Hindi Corporate and Commercial Dialogues

## Overview
This project implements a **Multimodal Intent Recognition Framework** designed specifically for interpreting Hindi and Hinglish dialogues in corporate settings and commercial scenarios (e.g., grocery shops, office encounters). By leveraging multiple data modalities—**Video (Visuals), Audio (Acoustics), and Text (Transcripts)**—the model accurately captures and classifies the underlying intents behind conversational utterances.

## Core Methodology & References
The underlying architecture and feature extraction strategies are built upon the foundational research provided in:
- **`MintRec_capstone.pdf`**: Guides the overall multimodal fusion strategy, establishing the baseline pipeline for combining text, audio, and visual streams to improve intent classification accuracy.
- **`Wavelet_Capstone.pdf`**: Informs the advanced feature extraction techniques, utilizing Wavelet transforms to better capture time-frequency representations (particularly vital for nuanced acoustic and visual signal processing).

## Dataset Structure
The system is trained and evaluated on a custom dataset tailored to corporate life and bazaar (marketplace) scenes.
- **`Corporate_dataset.xlsx`**: The primary dataset registry containing metadata, textual transcripts (Hindi and Hinglish), timestamps, and video mapping.
- **Video Mapping**: The respective video inputs are mapped directly via their **Video IDs**, which are uniquely composed of the scene name combined with the corresponding utterance number.
- **Modality Details**:
  - **Text**: Hindi and Hinglish transcripts.
  - **Audio**: Raw audio waveforms.
  - **Video**: Sequential frames capturing subject behavior and environmental context.

## Intent Classes
The categorization taxonomy is strictly defined in **`Intent_class.xlsx`**. This includes a broad spectrum of intents relevant to Indian corporate and commercial interactions, enabling the model to distinguish between standard business communications, negotiations, casual banter, and potentially anomalous behavior (like blackmail or gafla/scams).


## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch / TensorFlow
- Pandas (for dataset aggregation)
- FFmpeg (for Audio/Video processing)
- Hugging Face `transformers` (for NLP feature extraction)

### Installation & Usage
1. Clone the repository and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the `Corporate_dataset.xlsx` and `Intent_class.xlsx` files are in the root directory alongside the video category folders.
3. Run the data preprocessing script to align modalities:
   *(Scripts like `process_videos.py` can be executed to parse the metadata and prepare the input tensors).*

---

## References
- **MintRec Framework**: [arXiv:2506.10011](https://arxiv.org/abs/2506.10011)
- **Wavelet-based Multimodal Representation**: [arXiv:2403.10943](https://arxiv.org/abs/2403.10943)

*Developed as part of the Capstone research encompassing MintRec and Wavelet-based Multimodal Intent Recognition.*
