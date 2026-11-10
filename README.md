# Automated Synthetic Speech Quality Assessment (UTMOSv2) 🎙️📈

An end-to-end framework for automatic Mean Opinion Score (MOS) prediction of synthetic speech (Text-to-Speech & Voice Conversion) using the **UTMOSv2** deep learning architecture in acoustic-only (`spec_only`) mode. 

This project encompasses local CPU inference, model training & fine-tuning on the **MetaCentrum** HPC cluster, cross-domain/multilingual evaluations, and a standalone **FastAPI + Vanilla JS** local web application for real-time audio evaluation.

---

## 📌 Project Overview & Goals

Human evaluation remains the gold standard for speech quality assessment, but it is expensive, time-consuming, and difficult to scale. This project leverages **UTMOSv2** (*UTokyo-SaruLab Mean Opinion Score System v2*), a self-supervised learning network trained to estimate perceptual MOS scores (ranging from `1.00` to `5.00`) directly from raw audio waveforms and spectrograms.

### Key Technical Highlights
- **Purely Acoustic Mode (`spec_only`)**: Operates without text transcriptions or phoneme alignments, ensuring universal language applicability.
- **HPC Cluster Workflows**: Configured and executed 20-epoch training and 5-epoch fine-tuning pipelines on MetaCentrum GPU nodes.
- **Compatibility & Bug Fixes**: Resolved Windows OS downloading limitations (`urllib.request` patch) and tensor dimension mismatches (`fc.weight` layer alignment).
- **Cross-Domain & Multilingual Benchmarking**: Evaluated generalization capabilities across single-language (English) and multi-language (Chinese, Shanghainese, Spanish, French, English) datasets.
- **Local Web Interface**: Privacy-first, zero-external-dependency web application providing real-time MOS score estimation (~2 seconds per file).

---

## 📊 Datasets & Experimental Setup

### 1. VoiceMOS Challenge 2022 Dataset
A single-language (English) dataset comprising **7,106 audio files** collected from various TTS and VC models with crowdsourced human MOS annotations.
- **Train Set**: 4,974 samples
- **Validation Set**: 1,066 samples
- **Test Set**: 1,066 samples

### 2. BC-SOMOS (Samsung Open MOS Dataset)
A multilingual dataset comprising **2,731 audio files** across 5 languages (Chinese, Shanghainese, Spanish, French, English) used to evaluate zero-shot domain adaptation and model fine-tuning.
- **Train Set**: 1,901 samples
- **Validation Set**: 101 samples
- **Test Set**: 729 samples

---

## 🛠️ Technical Implementation & Workarounds

### 1. Windows Download Patch (`_download.py`)
Standard UTMOSv2 relies on `wget` system calls for pulling pretrained Hugging Face weights, which fails natively on Windows. The utility was patched using Python's native `urllib.request`:
```python
import urllib.request

def download_file(url, destination):
    urllib.request.urlretrieve(url, destination)
```

### 2. Dimension Mismatch Fix (`model_creations.ipynb`)
Standard pretrained UTMOSv2 models expect output dimensions of `[1, 5130]` for multi-task classification across 7 datasets. Training exclusively in `spec_only` mode produces a 5,123-dimension final linear output (`fc.weight`), causing a `RuntimeError: size mismatch`. 

A conversion script instantiates a zero-padded tensor `[1, 5130]` and copies the trained weights into place, allowing seamless evaluation via the official `inference.py` utility.

### 3. HPC MetaCentrum Execution
Training was offloaded to MetaCentrum GPU clusters using interactive Jupyter Lab sessions. CUDA kernel errors on older architectures (GTX 1080 Ti) were bypassed by explicitly excluding incompatible GPU nodes.

---

## 📈 Benchmark Results & Evaluation Metrics

Performance is evaluated at both the **Utterance-level** (individual file accuracy) and **System-level** (aggregated score per TTS/VC engine) using four standard metrics:
- **MSE** (Mean Squared Error) ↓ *(Ideal: 0)*
- **LCC** (Pearson Linear Correlation Coefficient) ↑ *(Ideal: 1.0)*
- **SRCC** (Spearman Rank Correlation Coefficient) ↑ *(Ideal: 1.0)*
- **KTAU** (Kendall Tau Rank Correlation) ↑ *(Ideal: 1.0)*

### 1. Primary Model Evaluation (VoiceMOS 2022 Test Set)
*Model trained for 20 epochs across 2 cross-validation folds.*

| Evaluation Metric | Utterance-level | System-level |
| :--- | :---: | :---: |
| **MSE** ↓ | **0.2966** | **0.2042** |
| **LCC** ↑ | **0.8105** | **0.8484** |
| **SRCC** ↑ | **0.8124** | **0.8453** |
| **KTAU** ↑ | **0.6255** | **0.6654** |

### 2. Zero-Shot Generalization Test (BC-SOMOS Test Set)
*Direct evaluation on multilingual BC-SOMOS before fine-tuning.*

| Evaluation Metric | Utterance-level | System-level |
| :--- | :---: | :---: |
| **MSE** ↓ | 0.5385 | 0.5402 |
| **LCC** ↑ | 0.5100 | 0.5091 |
| **SRCC** ↑ | 0.4771 | 0.4774 |
| **KTAU** ↑ | 0.3302 | 0.3287 |

### 3. Fine-Tuning Analysis (BC-SOMOS Dataset)
*Fine-tuned for 5 epochs with a reduced learning rate ($lr = 2 \times 10^{-5}$, $batch\_size = 8$).*

| Benchmark Subset | Utterance MSE ↓ | Utterance LCC ↑ | System MSE ↓ | System LCC ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **BC-SOMOS (Train)** | 0.2474 | 0.7499 | 0.2479 | 0.7505 |
| **BC-SOMOS (Test)** | 0.7431 | 0.4349 | 0.7431 | 0.4349 |
| **VoiceMOS 2022 (Retest)** | **0.4339** | **0.7246** | **0.2700** | **0.8030** |

> **Key Finding:** While fine-tuning on the small BC-SOMOS dataset exhibited overfitting to the training split, cross-domain retesting on VoiceMOS 2022 confirmed that the underlying UTMOSv2 feature extractor retained strong voice evaluation stability (System LCC remaining high at `0.8030`).

---

## 🌐 Local Web Interface (FastAPI + HTML/CSS/JS)

A lightweight client-server application for local, privacy-centric voice quality estimation.

```
+-------------------+      HTTP POST (.wav)      +--------------------+
|                   | -------------------------> |                    |
| Web Browser UI    |                            | FastAPI Server     |
| (HTML/CSS/JS)     | <------------------------- | (UTMOSv2 In-Mem)   |
+-------------------+     Predicted MOS (1-5)    +--------------------+
```

### Features & Workflow
1. **Interactive Upload**: Drag & drop or select `.wav` files directly in the browser.
2. **Audio Preview**: Integrated web player to listen to candidate audio files.
3. **Local Inference**: FastAPI routes the file buffer to PyTorch in CPU mode (`num_workers=0`).
4. **Immediate Feedback**: Display rendered output score within ~5 seconds.

---

## 🚀 Getting Started & Execution

### Prerequisites
- Python 3.10
- PyTorch 2.0+ with Torchaudio
- Anaconda / Miniconda

### 1. Environment Setup
```bash
git clone https://github.com/your-username/automated-speech-quality-utmosv2.git
cd automated-speech-quality-utmosv2

# Create and activate environment
conda create -n utmos_env python=3.10 -y
conda activate utmos_env

# Install UTMOSv2 and dependencies
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
pip install fastapi uvicorn librosa torchaudio jinja2 python-multipart
```

### 2. Running Local CPU Inference (Script)
To run MOS prediction over a directory of `.wav` files:
```python
import utmosv2

# Load model with local weights
model = utmosv2.create_model(pretrained=True)
results = model.predict(input_dir="./test_audio", device="cpu", num_workers=0)
print(results)
```

## 📜 References & Acknowledgments
1. **VoiceMOS Challenge 2022**: Wen-Chin Huang et al. ([arXiv:2203.11389](https://arxiv.org/abs/2203.11389))
2. **UTMOSv2**: Takaaki Saeki et al. ([arXiv:2409.09305](https://github.com/sarulab-speech/UTMOSv2))
3. **BC-SOMOS Dataset**: Samsung Open MOS Dataset ([Zenodo:10691660](https://zenodo.org/records/10691660))
4. **Author**: Ondřej Rottenborn, Department of Cybernetics (KKY), Faculty of Applied Sciences, University of West Bohemia.
