# Energy Consumption Anomaly Detection – TU Dortmund Case Study

## Project Overview

This project focuses on **Time Series Anomaly Detection** in energy consumption data collected from buildings at TU Dortmund University. It compares multiple machine learning models to detect point and contextual anomalies in consumption patterns.

## Problem Statement

Energy consumption data is critical for identifying inefficiencies and reducing carbon footprints.  
Anomalies in such data may indicate:

- Equipment malfunctions
- Abnormal usage patterns
- Sensor errors or data quality issues

Accurate anomaly detection helps organizations take **proactive measures to improve operational efficiency and sustainability**.

### Goal:

- Detect anomalies in time series energy consumption data
- Compare performance of three deep learning models:
    - **LSTM Autoencoder (LSTM-AE)**
    - **Temporal Convolutional Autoencoder (TCN-AE)**
    - **Deep Generative Hierarchical Latent Model (DGHL)**

## Models Implemented

### 1️⃣ LSTM Autoencoder (LSTM-AE)

- **Architecture:** Encoder-Decoder using LSTM layers
- **Anomaly Detection:** Based on reconstruction error
- **Reference:** Malhotra et al. (2016)

### 2️⃣ Temporal Convolutional Network Autoencoder (TCN-AE)

- **Architecture:** Dilated causal convolutions for wide receptive fields
- **Anomaly Detection:** Based on reconstruction error
- **Reference:** Thill et al. (2021)

### 3️⃣ Deep Generative Model with Hierarchical Latent Factors (DGHL)

- **Architecture:** Top-down ConvNet with hierarchical latent factors
- **Training:** MCMC sampling and alternating backpropagation
- **Anomaly Detection:** High reconstruction error indicates anomalies
- **Reference:** Challu et al. (2022)

## Dataset

**Data Source:** TU Dortmund University – 10 campus buildings

**Note:**  
Due to privacy concerns, **the dataset is not included in this repository**. The code is designed to work with similar time series datasets containing energy consumption data.

## Visualizations

**Sample output plots** are saved in the `outputs/` folder.

## Challenges

### Data Quality

- Real-world sensor data had missing values and noise.

### Labeling Limitations

- True anomaly labels were not always available.
- Public holidays were used as proxy anomalies for validation.

## Learnings & Contributions

- Hands-on experience with deep learning models for time series data.
- Explored trade-offs between recurrent (LSTM) and convolutional (TCN) architectures.
- Implemented advanced generative modeling with DGHL, learning hierarchical representations.
- Team collaboration on large-scale ML pipelines and experimental evaluations.

## Authors

**TU Dortmund University – Case Study Team**

- Neena Sharon Betti Mol
- Rahul Poovassery
- Akash Chandra Baidya
- Mridul Varghese Koshy
- Nidhi Patel
- Ashish Saini
- Kübra Turum
- Supritha Palguna
- Akanksha Tanwar

## License

This project is for **educational and academic purposes only**.

---
