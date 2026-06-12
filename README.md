Ethical AI Toxicity Detection System

Overview

This project develops a fairness-aware toxicity detection framework using multiple machine learning and deep learning models. The system combines the Jigsaw Toxic Comment Dataset, Hate Speech Dataset, and Civil Comments Dataset to identify toxic content while evaluating fairness and bias.

Features

- Toxicity Detection using Logistic Regression
- Model Comparison (Logistic Regression, Naive Bayes, SVM, Random Forest)
- BERT-based Toxicity Classification
- BiLSTM-based Toxicity Classification
- Fairness Evaluation
- Statistical Validation using Cross-Validation and T-Test
- Explainable AI using LIME and SHAP
- Interactive Toxicity Prediction System

---

Project Structure

ethical_ai/
│
├── dataset/
│   ├── jigsaw_cleaned.csv
│   ├── hate_cleaned.csv
│   ├── civil_cleaned.csv
│   └── final_combined_dataset.csv
│
├── trainlevel1.py          # Logistic Regression + Fairness
├── trainlevel2.py          # Model Comparison
├── train_bert.py           # BERT Model
├── train_bilstm.py         # BiLSTM Model
├── new_predict.py          # Interactive Prediction System
├── lime_explain.py         # LIME Explainability
├── shap_waterfall.py       # SHAP Waterfall Plot
├── shap_beeswarm.py        # SHAP Beeswarm Plot
│
├── lr_model.pkl
├── tfidf.pkl
└── README.md

---

Installation

Clone Repository

git clone https://github.com/pavansmca24/ethical_ai.git
cd ethical_ai

Create Virtual Environment

python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux/Mac

source venv/bin/activate

Install Dependencies

pip install pandas numpy scikit-learn scipy joblib
pip install transformers torch
pip install tensorflow
pip install lime shap
pip install matplotlib

Or:

pip install -r requirements.txt

---

Dataset Preparation

## Dataset Preparation

The datasets used in this project are hosted on Google Drive due to GitHub storage limitations.

### Download Datasets

Google Drive Link:

https://drive.google.com/drive/folders/1D-NcHBWJu1T92Ik1bluHUmbxkXr1Z5JZ?usp=drive_link

After downloading, place the files inside the `dataset/` folder:

dataset/
├── jigsaw_cleaned.csv
├── hate_cleaned.csv
├── civil_cleaned.csv
└── final_combined_dataset.csv

### Datasets Used

1. Jigsaw Toxic Comment Dataset
2. Hate Speech and Offensive Language Dataset
3. Civil Comments Dataset

## Note

Due to GitHub file size limitations, datasets, trained models, and checkpoints are not stored in this repository. They can be downloaded from the Google Drive link provided above.

---

Running the Models

1. Logistic Regression + Fairness Evaluation

python trainlevel1.py

Outputs:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Demographic Parity
- Equalized Odds
- Cross Validation Results
- T-Test Results

Saved Files:

lr_model.pkl
tfidf.pkl

---

2. Model Comparison

python trainlevel2.py

Models:

- Logistic Regression
- Naive Bayes
- SVM
- Random Forest

Outputs:

- Accuracy
- ROC-AUC
- Confusion Matrix
- Classification Report

---

3. BERT Model

python train_bert.py

Outputs:

- Accuracy
- Precision
- Recall
- F1 Score

Uses:

bert-base-uncased

---

4. BiLSTM Model

python train_bilstm.py

Outputs:

- Accuracy
- Classification Report

Architecture:

Embedding
→ BiLSTM
→ Dropout
→ Dense
→ Sigmoid

---

Interactive Toxicity Prediction

After training Logistic Regression:

python new_predict.py

Example:

Enter text: You are stupid

Output:
Toxic (confidence: 0.91)

---

Explainable AI

LIME

python lime_explain.py

Output:

lime_explanation.html

Used to explain individual predictions.

---

SHAP Waterfall Plot

python shap_waterfall.py

Generates a local explanation for a single prediction.

---

SHAP Beeswarm Plot

python shap_beeswarm.py

Generates a global feature importance visualization across multiple samples.

---

Fairness Evaluation

The system evaluates fairness using:

Demographic Parity

Measures whether positive predictions are distributed fairly across samples.

Equalized Odds

Measures the True Positive Rate across classes.

Bias Analysis

Identity-based examples are used to examine residual model bias.

Examples:

She is Muslim
She is Black
I respect Muslims
I hate Muslims

---

Technologies Used

- Python
- Scikit-learn
- TensorFlow/Keras
- Hugging Face Transformers
- BERT
- BiLSTM
- LIME
- SHAP
- Pandas
- NumPy

---

Results

Example Results:

Model| Accuracy
Logistic Regression| ~93%
BiLSTM| ~93%
BERT| ~92–94%

The project demonstrates strong toxicity detection performance while incorporating fairness evaluation and explainability techniques.

---

Author

Pavan S

BMS College of Engineering

Ethical AI Toxicity Detection Project
