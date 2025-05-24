# Ensemble Deep Learning for Advanced EEG-Based Grasp-and-Lift Detection

## Overview

This project implements a novel ensemble deep learning framework for detecting grasp-and-lift (GAL) events from electroencephalography (EEG) signals. The approach is tailored for Brain-Computer Interface (BCI) applications, particularly for controlling prosthetic devices for individuals with neuromuscular impairments. By leveraging advanced signal processing, feature engineering, and a hybrid of deep learning architectures, this project achieves state-of-the-art performance on the challenging multi-class GAL detection task.

---

## Features

- **Comprehensive EEG Preprocessing**: Dual-stage filtering (bandpass and notch) and robust scaling to standardize and enhance signal quality.
- **Handcrafted Feature Extraction**: Time- and frequency-domain features, including Hjorth parameters, statistical measures, and power spectral densities.
- **Data Augmentation**: Techniques such as Gaussian noise injection, channel dropout, and Mixup to improve model generalization.
- **Deep Learning Models**:
  - **EEGNet-TCN**: Combines spatial filtering (EEGNet) with dilated temporal convolutions (TCN).
  - **CNN-BiLSTM**: Extracts spatial and temporal patterns using convolutional and bidirectional LSTM layers.
  - **Attention-Based Model**: Leverages multi-head self-attention for dynamic weighting of time steps and channels.
- **Ensemble Strategy**: Stacked meta-ensemble (gradient boosting regressor) with test-time augmentation for robust final predictions.
- **Extensive Validation**: Stratified cross-validation, detailed per-class metrics, and comparison with prior work.

---

## Dataset

- **WAY-EEG-GAL Dataset**: 32-channel EEG recordings from 12 healthy subjects performing grasp-and-lift tasks.
- **Sampling Rate**: 500 Hz.
- **Events Detected**:
  - Hand Start (HS)
  - First Digit Touch (FDT)
  - Both Start Load Phase (BSP)
  - Lift Off (LO)
  - Replace (R)
  - Both Released (BR)

> For more details, see the original [Kaggle competition](https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection).

---

## Methodology

1. **Signal Preprocessing**:
    - Butterworth bandpass (0.5–60 Hz)
    - Notch filter (50 Hz)
    - Robust scaling (median, IQR)

2. **Feature Engineering**:
    - Time-domain: Mean, std, Hjorth parameters
    - Frequency-domain: Relative band power (Delta, Theta, Alpha, Beta, Gamma)

3. **Data Augmentation**:
    - Gaussian noise
    - Channel dropout
    - Mixup (sample interpolation)

4. **Model Architectures**:
    - EEGNet-TCN
    - CNN-BiLSTM
    - Attention-based network

5. **Ensemble Learning**:
    - 5-fold cross-validation
    - Test-time augmentation
    - Meta-ensemble with gradient boosting

---

## Results

| Class | Accuracy | Precision | Recall | F1 Score | AUC    |
|-------|----------|-----------|--------|----------|--------|
| HS    | 0.9132   | 0.9299    | 0.6933 | 0.7944   | 0.9768 |
| FDT   | 0.9810   | 0.9945    | 0.9280 | 0.9601   | 0.9990 |
| BSP   | 0.9881   | 0.9882    | 0.9625 | 0.9752   | 0.9994 |
| LO    | 0.9546   | 0.9267    | 0.8852 | 0.9055   | 0.9925 |
| R     | 0.9434   | 0.9356    | 0.8255 | 0.8771   | 0.9889 |
| BR    | 0.9416   | 0.9316    | 0.8253 | 0.8752   | 0.9871 |
| **Avg** | **0.9537** | **0.9511** | **0.8533** | **0.8979** | **0.9906** |

- **Average AUC**: 0.9906 (Ensemble)
- **Average Accuracy**: 0.9537 (Ensemble)
- Outperforms prior state-of-the-art models.

---

## Usage

### Installation

Clone the repository and install required dependencies (see `requirements.txt`):

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

### Training

Prepare the dataset as described above and run:

```bash
python train.py --config configs/ensemble_config.yaml
```

### Evaluation

Evaluate and visualize results:

```bash
python evaluate.py --model ensemble --data test
```

### Visualization

Generated figures (e.g., ROC curves, confusion matrices) can be found in the `figures/` directory after running evaluation.

---

## Project Structure

```
.
├── data/             # Raw and preprocessed EEG data
├── models/           # Model definitions (EEGNet, CNN-BiLSTM, Attention)
├── configs/          # Configuration files for training/evaluation
├── scripts/          # Preprocessing, feature extraction, augmentation
├── figures/          # Training curves and evaluation plots
├── train.py          # Main training script
├── evaluate.py       # Testing/evaluation script
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## References

- Lawhern et al., EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces, 2018.
- Bai et al., An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, 2018.
- Hasan et al., Deep Learning Approaches to EEG-based Grasp-and-Lift Detection, 2022.
- Xu et al., Deep Transfer Learning for EEG Signal Classification, 2019.
- [Kaggle: Grasp-and-Lift EEG Detection](https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection)

---

## Acknowledgements

- National Institute of Technology, Delhi
- Supervising faculty and contributing authors
- Open-source contributors in EEG and BCI research

---

## License

This project is for academic and research purposes only. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or contributions, please contact:

- Aditya Shaurya Singh Negi — 221210012@nitdelhi.ac.in
- Akshat Singh — 221210015@nitdelhi.ac.in
- Akshat — 221210014@nitdelhi.ac.in
- Dr. Arjun Singh Rawat — arjunsinghrawat005@gmail.com
