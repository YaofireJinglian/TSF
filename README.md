<div align="center">

# CDTF-Mamba
### Cross-Domain Time-Frequency Mamba for Long-Term Time Series Forecasting

[![Paper](https://img.shields.io/badge/Paper-Knowledge%20Based%20Systems%202026-b31b1b.svg)](#)
[![Journal](https://img.shields.io/badge/Venue-Knowledge%20Based%20Systems-1f6feb.svg)](https://www.sciencedirect.com/journal/knowledge-based-systems)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Official implementation of**
[*Cross-Domain Time-Frequency Mamba: A More Effective Model for Long-Term Time Series Forecasting*](#) **(Knowledge-Based Systems 2026)**

</div>

---

## 📌 Overview

**CDTF-Mamba** is a novel Mamba-based framework for **Long-Term Time Series Forecasting (LTSF)** that jointly models temporal dynamics in both the **time** and **frequency** domains.

Unlike conventional single-domain forecasting methods, CDTF-Mamba captures both **local temporal fluctuations** and **global periodic dependencies** through a collaborative cross-domain modeling paradigm. By integrating hierarchical temporal decomposition with frequency-aware state evolution, the model achieves superior forecasting accuracy while maintaining the efficiency and linear complexity advantages of the Mamba architecture.

<p align="center">
  <img src="Overview.png" alt="CDTF-Mamba Framework" width="90%"/>
  <br/>
  <em>Figure 1. Overall framework of CDTF-Mamba. The model simultaneously models time-domain and frequency-domain representations. The Time-domain Pyramid Mamba (TPM) captures multi-scale local dependencies, while the Frequency-domain Decomposition Mamba (FDM) extracts global periodic patterns and mitigates non-stationarity, enabling robust long-term forecasting.</em>
</p>

---

## 🎯 Key Features

- ⏳ **Time-domain Pyramid Mamba (TPM)** — captures multi-scale temporal dependencies and local fluctuations through hierarchical temporal modeling.
- 🌊 **Frequency-domain Decomposition Mamba (FDM)** — models periodic structures and stabilizes sequence evolution in the frequency domain.
- 🔄 **Cross-Domain Fusion Mechanism** — enables effective interaction between temporal and spectral representations for comprehensive sequence understanding.
- ⚡ **Linear Complexity Forecasting** — inherits the efficiency and scalability advantages of the Mamba architecture.
- 📊 **Comprehensive Benchmark Evaluation** — validated on multiple real-world datasets spanning traffic, energy, weather, and industrial forecasting tasks.

---

## 📊 Results Highlights

| Metric | Result |
|--------|--------|
| **Forecasting Accuracy** | Achieves SOTA performance on major benchmarks |
| **Long-Term Dependency Modeling** | Superior temporal representation capability |
| **Efficiency** | Linear complexity with fast inference speed |
| **Scalability** | Robust performance across varying prediction horizons |
| **Cross-Domain Modeling** | Effectively captures both local and global patterns |

---

## 📈 Datasets

The experiments are conducted on widely used long-term forecasting benchmarks:

- **ETTh1 / ETTh2**
- **ETTm1 / ETTm2**
- **Weather**
- **Traffic**
- **Electricity**
- **Exchange**
- **Solar-Energy**
- **PEMS**

All datasets are publicly available for research purposes.

---

## 🛠 Installation & Usage

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NVIDIA GPU (tested on RTX 4090)

### Install Dependencies

```bash
pip install -r requirements.txt
### Training
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
### Testing
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

## 📚 Citation

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
