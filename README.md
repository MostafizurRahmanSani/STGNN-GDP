# STGNN-GDP: Modeling Cross-Country Economic Dynamics for Multi-Horizon GDP Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

This repository implements multiple approaches for **GDP prediction** using:
- **STGNN** (Spatio-Temporal Graph Neural Network) - Leverages trade relationships between countries
- **ARIMA** - Classical time series forecasting baseline  
- **GRU** - Deep learning time series baseline

The models predict GDP for 199 countries across 3-year horizons using historical data (1996-2019) and international trade networks.

## 📊 Key Features

- **Graph-based modeling**: Incorporates real trade relationships between countries
- **Multi-horizon prediction**: Forecasts GDP for t+1, t+2, and t+3 years
- **Reproducible research**: Fixed seeds and deterministic algorithms
- **Comprehensive evaluation**: Multiple metrics (MSE, MAE, RMSE, R², accuracy)
- **Visualization tools**: Training curves, prediction scatter plots, attention heatmaps

## 🏗️ Project Structure

```
├── config.py                 # Global configuration & reproducibility settings
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── gru_model.pt             # Trained GRU model weights
├── arima_model.pkl         # Saved ARIMA model weights
├── stgnn_current.pt          # Trained STGNN model weights
├── data/                     # Data loading utilities
│   ├── __init__.py
│   └── data_loader.py       # STGNN dataset preparation
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── stgnn.py             # STGNN with message passing layers
│   ├── arima_model.py       # ARIMA training/evaluation
│   └── gru_model.py         # GRU model definition
├── training/                 # Training loops
│   ├── __init__.py
│   ├── train_stgnn.py       # STGNN training
│   ├── train_arima.py       # ARIMA training wrapper
│   └── train_gru.py         # GRU training wrapper
├── evaluation/               # Evaluation utilities
│   ├── __init__.py
│   ├── metrics.py           # Regression metrics
│   └── visualization.py     # Plotting functions
├── utils/                    # Helper utilities
│   ├── __init__.py
│   ├── helpers.py           # Seed setting
└── scripts/                  # Executable scripts
    ├── run_stgnn.py         # Train & evaluate STGNN
    ├── run_arima_gru.py     # Train ARIMA & GRU baselines
    ├── evaluate_stgnn.py    # Load & evaluate saved STGNN
    └── evaluate_baselines.py # Load & evaluate saved baselines
```

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/MostafizurRahmanSani/STGNN-GDP.git
cd STGNN-GDP

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train Models

```bash
# Train STGNN (takes ~30 minutes on GPU)
python -m scripts.run_stgnn

# Train ARIMA and GRU baselines
python -m scripts.run_arima_gru
```

### 3️⃣ Evaluate Saved Models

```bash
# Evaluate STGNN on test set
python -m scripts.evaluate_stgnn

# Evaluate ARIMA and GRU on test set
python -m scripts.evaluate_baselines
```

## 📈 Results

### Main Results: MAE Comparison (Primary Metric)

| Model | t+1 | t+2 | t+3 | Average |
|-------|-----|-----|-----|---------|
| ARIMA | 0.4137 | 0.4191 | 0.4378 | 0.4235 |
| GRU (temporal-only) | 0.4897 | 0.4941 | 0.5819 | 0.5219 |
| **STGNN (proposed)** | **0.2769** | **0.3032** | **0.4068** | **0.3289** |

**STGNN reduces average MAE by 22.3% relative to ARIMA and 37.0% relative to GRU.**

---

### Detailed Results: RMSE and R²

| Model | RMSE (t+1) | RMSE (t+2) | RMSE (t+3) | R² (t+1) | R² (t+2) | R² (t+3) |
|-------|------------|------------|------------|----------|----------|----------|
| ARIMA | 0.4857 | **0.4892** | **0.5140** | 0.9564 | **0.9555** | **0.9507** |
| GRU | 0.6273 | 1.1707 | 1.9135 | 0.9779 | 0.9263 | 0.8262 |
| **STGNN** | **0.4229** | 1.0517 | 1.8443 | **0.9899** | 0.9405 | 0.8386 |

STGNN achieves the **best RMSE and R² at t+1**, while ARIMA remains competitive at longer horizons—highlighting complementary strengths.


## 🔧 Configuration

All key parameters are centralized in `config.py`:

```python
PAST_WINDOW = 5      # Years of history used
HORIZON = 3          # Years to predict
FIRST_YEAR = 1996    # Start year
LAST_YEAR = 2019     # End year
TRAIN_END = 2009     # Train/val split
VAL_END = 2012       # Val/test split
```

## 🧪 Reproducibility

This project ensures **100% reproducible results** through:

- Fixed random seeds (`set_seed(42)`)
- Deterministic CUDA algorithms
- Environment variables for CuBLAS
- No hidden randomness in data loading

```python
from config import set_seed
set_seed(42)  # Same results every run
```

## 📊 Visualizations

The repository generates several plots:

| Plot | Description | Generated By |
|------|-------------|--------------|
| Training curves | MSE/MAE vs epochs | `run_stgnn.py` |
| Prediction scatter | Actual vs predicted | `run_stgnn.py` |
| ARIMA model selection | Order comparison | `run_arima_gru.py` |
| GRU training curves | Loss over time | `run_arima_gru.py` |

## 📚 Dataset

The project uses data from the [gnns_for_gdp](https://github.com/pboennig/gnns_for_gdp) repository:

- **199 countries and territories** in the trade network
- **171 countries** have complete GDP data for all years
- **1996-2019** yearly observations
- **Node features**: Population, CPI, Employment, Lagged GDP
- **Edge features**: 10-dimensional trade relationship vectors

## 🖥️ System Requirements

- **Python**: 3.8 or higher
- **CUDA**: Optional but recommended for STGNN training
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 500MB for data and models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{STGNN-GDP,
  author = {Mostafizur Rahman Sani},
  title = {STGNN-GDP: Modeling Cross-Country Economic Dynamics for Multi-Horizon GDP Forecasting},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MostafizurRahmanSani/STGNN-GDP}}
}
```

## 📧 Contact

Mostafizur Rahman Sani - sani.rahman0191@gmail.com

Project Link: [https://github.com/MostafizurRahmanSani/STGNN-GDP](https://github.com/MostafizurRahmanSani/STGNN-GDP)

---

**⭐ Star this repository if you find it useful!**

