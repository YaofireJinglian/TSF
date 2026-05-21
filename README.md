<div align="center">

# CDTF-Mamba
### Cross-Domain Time-Frequency Mamba for Long-Term Time Series Forecasting

[![Paper](https://img.shields.io/badge/Paper-Knowledge%20Based%20Systems%202026-b31b1b.svg)](#)
[![Journal](https://img.shields.io/badge/Venue-Knowledge%20Based%20Systems-1f6feb.svg)](https://www.sciencedirect.com/journal/knowledge-based-systems)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

**Official implementation of**  
[*Cross-Domain Time-Frequency Mamba: A More Effective Model for Long-Term Time Series Forecasting*]([#](https://www.sciencedirect.com/science/article/pii/S0950705126000845))  
**Knowledge-Based Systems (2026)**

</div>

---

# 📌 Overview

**CDTF-Mamba** is a novel Mamba-based framework for **Long-Term Time Series Forecasting (LTSF)** that jointly models temporal dynamics in both the **time domain** and **frequency domain**.

Unlike conventional single-domain forecasting methods, CDTF-Mamba effectively captures both:

- **Local temporal fluctuations**
- **Global periodic dependencies**

through a collaborative **cross-domain time-frequency modeling paradigm**.

The framework integrates:

- **Time-domain Pyramid Mamba (TPM)** for hierarchical temporal dependency modeling
- **Frequency-domain Decomposition Mamba (FDM)** for spectral evolution learning and non-stationary sequence stabilization
- **Cross-domain fusion mechanisms** for enhanced representation interaction

As a result, CDTF-Mamba achieves superior long-term forecasting performance while maintaining the **linear complexity** and **efficient inference capability** inherited from the Mamba architecture.

---

# 🧠 Framework

<p align="center">
  <img src="Overview.png" alt="CDTF-Mamba Framework" width="90%"/>
</p>

<p align="center">
  <em>
  Figure 1. Overall framework of CDTF-Mamba. The model jointly learns time-domain and frequency-domain representations. TPM captures multi-scale local temporal dependencies, while FDM extracts global periodic patterns and mitigates non-stationarity, enabling robust long-term forecasting.
  </em>
</p>

---

# 🎯 Key Features

## ⏳ Time-domain Pyramid Mamba (TPM)

- Hierarchical temporal modeling
- Multi-scale dependency extraction
- Enhanced local fluctuation perception
- Progressive receptive field expansion

## 🌊 Frequency-domain Decomposition Mamba (FDM)

- Frequency-aware state evolution
- Global periodic pattern modeling
- Spectral decomposition learning
- Improved robustness to non-stationary sequences

## 🔄 Cross-Domain Fusion

- Dynamic interaction between temporal and spectral representations
- Complementary feature enhancement
- More comprehensive sequence understanding

## ⚡ Efficient Forecasting

- Linear computational complexity
- Fast inference speed
- Memory-efficient sequence modeling
- Scalable to long forecasting horizons

---

# 📊 Experimental Results

CDTF-Mamba is extensively evaluated on multiple real-world long-term forecasting benchmarks.

| Evaluation Aspect | Performance |
|------------------|-------------|
| Forecasting Accuracy | Achieves state-of-the-art performance |
| Long-Term Dependency Modeling | Strong temporal representation capability |
| Efficiency | Fast inference with linear complexity |
| Scalability | Stable across different prediction horizons |
| Cross-Domain Modeling | Captures both local and global patterns effectively |

---

# 📈 Supported Datasets

The experiments are conducted on widely used long-term forecasting benchmarks:

- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Weather
- Traffic
- Electricity
- Exchange
- Solar-Energy
- PEMS

All datasets are publicly available for research purposes.

---

# 🛠 Installation

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+
- NVIDIA GPU (tested on RTX 4090)

## Clone Repository

```bash
git clone https://github.com/yourname/CDTF-Mamba.git
cd CDTF-Mamba
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

```bash
python main.py \
    --is_training 1 \
    --model_id "CDTF_Mamba_test" \
    --model "CDTF-Mamba" \
    --data "PEMS" \
    --root_path "./data/PEMS" \
    --data_path "PEMS08.npz" \
    --seq_len 96 \
    --pred_len 12 \
    --enc_in 170 \
    --num_layers 2 \
    --n1 512 \
    --d_state 256 \
    --dconv 2 \
    --e_fact 1 \
    --k 3 \
    --ch_ind 1 \
    --revin 1 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --use_gpu True \
    --gpu 0
```

---

# 🧪 Testing

```bash
python main.py \
    --is_training 0 \
    --model_id "CDTF_Mamba_test" \
    --model "CDTF-Mamba" \
    --data "PEMS" \
    --root_path "./data/PEMS" \
    --data_path "PEMS08.npz" \
    --seq_len 96 \
    --pred_len 12 \
    --enc_in 170 \
    --num_layers 2 \
    --n1 512 \
    --d_state 256 \
    --k 3 \
    --ch_ind 1 \
    --revin 1 \
    --use_gpu True \
    --gpu 0
```

---

# 📂 Project Structure

```bash
CDTF-Mamba/
│── data/
│── models/
│── layers/
│── utils/
│── scripts/
│── checkpoints/
│── main.py
│── requirements.txt
└── README.md
```

---

# 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{duan2026cross,
  title={Cross-Domain Time-Frequency Mamba: A More Effective Model for Long-Term Time Series Forecasting},
  author={Duan, Yuhang and Lin, Lin and Liu, Jinyuan and Zhang, Qing and Fan, Xin},
  journal={Knowledge-Based Systems},
  pages={115341},
  year={2026},
  publisher={Elsevier}
}
```

---

# ⭐ Acknowledgement

This repository is built upon several excellent open-source time series forecasting projects and Mamba-based sequence modeling frameworks. We sincerely thank the authors for their valuable contributions to the research community.



---
