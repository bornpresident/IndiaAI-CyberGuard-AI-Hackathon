# Cybercrime Classification System 🛡️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)]()
[![BERT](https://img.shields.io/badge/BERT-base-yellow)]()

## Overview
Hierarchical BERT-based system for classifying cybercrime complaints into categories and subcategories. Achieves 89.2% accuracy in multi-label classification.

## Key Results 📊
- Accuracy: 89.2%
- Precision: 87.6%
- Recall: 86.9%
- F1 Score: 87.2%

### Category Performance
Financial Fraud:  91% accuracy
Cyber Attack:    88% accuracy
Social Media:    87% accuracy

## Features ⭐
- Multi-lingual support (Hindi-English)
- Hierarchical classification
- Custom BERT architecture
- Real-time prediction capabilities

## Quick Start 🚀
```bash
# Clone repository
git clone https://github.com/username/cybercrime-classification.git

# Install dependencies
pip install -r requirements.txt

# Train model
python src/main.py

# Make predictions
python src/predict.py
```
Directory Structure 📁
```
cybercrime_project/
├── data/               # Dataset files
├── src/               # Source code
├── models/            # Trained models
└── results/           # Evaluation results
```
