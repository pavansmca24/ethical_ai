import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.utils import resample


print("LEVEL 2 STARTED ")


# -------------------------
# LOAD DATASETS
# -------------------------
print("Loading datasets...")

jigsaw = pd.read_csv("jigsaw_cleaned.csv")
hate = pd.read_csv("hate_cleaned.csv")
civil = pd.read_csv("civil_cleaned.csv")

print("Jigsaw:", jigsaw.shape)
print("Hate:", hate.shape)
print("Civil:", civil.shape)


# -------------------------
# REDUCE SIZE (FAST RUN)
# -------------------------
jigsaw = jigsaw.sample(5000, random_state=42)
hate = hate.sample(5000, random_state=42)
civil = civil.sample(5000, random_state=42)

print("Sampling done ")


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train_models(data, dataset_name):

    print(f"\n==================== {dataset_name} ====================")

    X = data['text']
    y = data['label']

    print("Original class distribution:\n", y.value_counts())

    # -------------------------
    # SPLIT FIRST (NO LEAKAGE)
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # BALANCE TRAINING DATA
    # -------------------------
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})

    df_majority = train_df[train_df.label == 0]
    df_minority = train_df[train_df.label == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    train_balanced = pd.concat([df_majority, df_minority_upsampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42)

    X_train = train_balanced['text']
    y_train = train_balanced['label']

    print("Balanced training distribution:\n", y_train.value_counts())

    # -------------------------
    # VECTORIZATION
    # -------------------------
    print("Vectorizing text...")

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Vectorization done ")

    # -------------------------
    # MODELS (LEVEL 2)
    # -------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():

        print(f"\n--- Training {name} ---")

        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        print("Accuracy:", accuracy_score(y_test, preds))

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_vec)[:, 1]
            print("ROC-AUC:", roc_auc_score(y_test, probs))
        else:
            print("ROC-AUC: Not available")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))


# -------------------------
# RUN TRAINING
# -------------------------
train_models(jigsaw, "Jigsaw Dataset")
train_models(hate, "Hate Speech Dataset")
train_models(civil, "Civil Comments Dataset")