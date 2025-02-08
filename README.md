# GAN-Isolation Forest for Stock Market Anomaly Detection

This project implements a hybrid anomaly detection system using Generative Adversarial Networks (GANs) and Isolation Forest on stock market data. The goal is to identify unusual patterns in stock prices that may indicate fraudulent activities or significant market events.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction
Stock market anomalies can indicate fraudulent activities, economic crises, or significant market shifts. This project applies wavelet transform, LSTM-based GANs, and Isolation Forest to detect these anomalies in stock price movements.

## Dataset
We use Yahoo Finance stock data for training and testing. Specifically, the project downloads historical stock data for `SADHNA.BO` from January 1, 2022, to January 1, 2023.

## Methodology
1. **Data Preprocessing**:
   - Missing values are handled using forward fill.
   - A 20-day moving average (MA20) is added as a feature.
   - MinMax scaling is applied to normalize the dataset.
   
2. **Wavelet Transform**:
   - Wavelet decomposition is performed using the 'db4' wavelet to extract meaningful features from the stock data.
   
3. **GAN Training**:
   - A GAN with LSTM layers is implemented to generate synthetic sequences.
   - The discriminator evaluates whether a sequence is real or generated.
   - The generator learns to create sequences resembling real data.

4. **Anomaly Detection**:
   - Reconstruction error from the trained generator is computed.
   - Isolation Forest is used to detect anomalies based on learned data patterns.

5. **Visualization**:
   - Stock prices are plotted along with detected anomalies for analysis.

## Installation
To run this project, install the following dependencies:
```bash
pip install numpy pandas matplotlib keras scikit-learn yfinance pywavelets
```

## Usage
Run the main script to train the GAN and detect anomalies:
```bash
python gan_isolation_forest.py
```
This script:
- Downloads stock data
- Preprocesses data and applies wavelet transform
- Trains GAN
- Detects anomalies using reconstruction error and Isolation Forest
- Visualizes results

## Results
The final output is a plot of stock prices with identified anomalies marked in red.

## Contributors
- @Kartikay45
- @19tanishq


## License
This project is open-source and available under the MIT License.

