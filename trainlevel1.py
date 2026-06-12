import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

from scipy.stats import ttest_1samp


print("SCRIPT STARTED")


# -------------------------
# LOAD DATA
# -------------------------
jigsaw = pd.read_csv("jigsaw_cleaned.csv")
hate = pd.read_csv("hate_cleaned.csv")
civil = pd.read_csv("civil_cleaned.csv")

for df in [jigsaw, hate, civil]:
    df['text'] = df['text'].fillna("")

# Increased dataset size (IMPORTANT)
jigsaw = jigsaw.sample(20000, random_state=42)
hate = hate.sample(20000, random_state=42)
civil = civil.sample(20000, random_state=42)

df = pd.concat([jigsaw, hate, civil])

print("Dataset:", df.shape)


# -------------------------
# SPLIT
# -------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------
# BALANCE TRAIN DATA
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

print("Balanced distribution:\n", y_train.value_counts())


# -------------------------
# VECTORIZATION
# -------------------------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -------------------------
# MODEL TRAINING
# -------------------------
model = LogisticRegression(max_iter=200, class_weight='balanced')

print("\nTraining Logistic Regression...")
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)


# -------------------------
# EVALUATION
# -------------------------
print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))


# -------------------------
# FAIRNESS METRICS
# -------------------------
def fairness_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    dp = np.mean(y_pred == 1)

    tpr = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)

    print("\nFairness Metrics:")
    print("Demographic Parity:", dp)
    print("Equalized Odds (TPR):", tpr)


fairness_metrics(y_test, preds)


# -------------------------
# CROSS VALIDATION
# -------------------------
print("\n--- Statistical Validation ---")

scores = cross_val_score(
    LogisticRegression(max_iter=200),
    X_train_vec,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Std Deviation:", scores.std())


# -------------------------
# T-TEST
# -------------------------
from scipy.stats import ttest_1samp

t_stat, p_value = ttest_1samp(scores, 0.5)

print("\nT-test Statistic:", t_stat)
print("P-value:", p_value)

# -------------------------
# SAVE MODEL (FOR LIME)
# -------------------------
joblib.dump(model, "lr_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("\nModel saved successfully")