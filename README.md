<div align="center">

# 📄 Deep Learning Paper Implementations

**From-scratch PyTorch implementations of landmark AI/ML research papers**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Papers](#-implemented-papers) • [Quick Start](#-quick-start) • [Structure](#-project-structure)

</div>

---

## 📌 Overview

Clean, well-documented **from-scratch implementations** of foundational deep learning papers. Each implementation includes detailed comments explaining the theory, annotated code mapping to equations in the paper, and training notebooks with visualizations.

### Goal

Demonstrate deep understanding of core AI architectures by implementing them from the ground up — no copy-pasting from libraries, just raw PyTorch.

---

## 📚 Implemented Papers

| # | Paper | Year | Key Contribution |
|---|-------|------|-----------------|
| 1 | **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** | 2017 | The Transformer architecture |
| 2 | **[An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)** | 2020 | Vision Transformer |
| 3 | **[BERT](https://arxiv.org/abs/1810.04805)** | 2018 | Bidirectional pre-training |
| 4 | **[GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** | 2018 | Generative pre-training |
| 5 | **[Denoising Diffusion (DDPM)](https://arxiv.org/abs/2006.11239)** | 2020 | Diffusion models |

---

## ✨ Key Features

- 🧠 **From Scratch** — Pure PyTorch, no high-level wrappers
- 📝 **Heavily Documented** — Line-by-line comments mapping to paper equations
- 📓 **Training Notebooks** — Jupyter notebooks for each paper with visualizations
- 🧪 **Unit Tested** — Every module has tests verifying correctness
- 📊 **Visualization** — Attention maps, training curves, and generated outputs

---

## 🚀 Quick Start

```bash
git clone https://github.com/altafpinjari2001/deep-learning-paper-implementations.git
cd deep-learning-paper-implementations

pip install -r requirements.txt

# Run Transformer training
python papers/transformer/train.py

# Or explore the notebooks
jupyter notebook notebooks/
```

---

## 📁 Project Structure

```
deep-learning-paper-implementations/
├── papers/
│   ├── transformer/           # Attention Is All You Need
│   │   ├── model.py           # Full Transformer implementation
│   │   ├── attention.py       # Multi-head attention
│   │   ├── layers.py          # Encoder/decoder layers
│   │   ├── train.py           # Training script
│   │   └── README.md          # Paper summary & notes
│   ├── vision_transformer/    # ViT
│   │   ├── model.py
│   │   ├── patch_embedding.py
│   │   └── train.py
│   ├── bert/                  # BERT
│   │   ├── model.py
│   │   └── pretrain.py
│   ├── gpt/                   # GPT
│   │   ├── model.py
│   │   └── generate.py
│   └── ddpm/                  # Diffusion Model
│       ├── model.py
│       ├── unet.py
│       └── sample.py
├── notebooks/
│   ├── 01_transformer.ipynb
│   ├── 02_vision_transformer.ipynb
│   └── 03_diffusion_models.ipynb
├── tests/
│   ├── test_transformer.py
│   └── test_vit.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center"><b>⭐ Star this repo if you find it useful!</b></div>
