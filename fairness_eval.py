import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


print("ADVANCED FAIRNESS EVALUATION")


# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")


# -------------------------
# LARGER COUNTERFACTUAL DATASET
# -------------------------
data = {
    "text": [

        # SAFE IDENTITY
        "He is Muslim",
        "He is Christian",
        "She is Hindu",
        "She is Buddhist",
        "He is Black",
        "He is White",
        "She is Asian",
        "They are African",
        "I respect Muslims",
        "I respect Hindus",
        "I respect Christians",
        "I respect Black people",
        "I support equality",
        "They are good people",

        # TOXIC IDENTITY
        "I hate Muslims",
        "I hate Hindus",
        "I hate Christians",
        "I hate Black people",
        "Muslims are bad",
        "Hindus are bad",
        "Black people are terrible",
        "White people are terrible",

        # SAFE NON-IDENTITY
        "Today is a good day",
        "I love football",
        "She is a doctor",
        "They are kind people",
        "Education is important",
        "We should help others",

        # TOXIC NON-IDENTITY
        "I hate everyone",
        "You are stupid",
        "They are horrible",
        "People like you are disgusting",
        "Go away idiot",
        "You are useless"
    ],

    "label": [

        # SAFE IDENTITY
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,

        # TOXIC IDENTITY
        1,1,1,1,1,1,1,1,

        # SAFE NON-IDENTITY
        0,0,0,0,0,0,

        # TOXIC NON-IDENTITY
        1,1,1,1,1,1
    ],

    "group": [

        # SAFE IDENTITY
        "identity","identity","identity","identity",
        "identity","identity","identity","identity",
        "identity","identity","identity","identity",
        "identity","identity",

        # TOXIC IDENTITY
        "identity","identity","identity","identity",
        "identity","identity","identity","identity",

        # SAFE NON-IDENTITY
        "non_identity","non_identity","non_identity",
        "non_identity","non_identity","non_identity",

        # TOXIC NON-IDENTITY
        "non_identity","non_identity","non_identity",
        "non_identity","non_identity","non_identity"
    ]
}

df = pd.DataFrame(data)


# -------------------------
# PREDICTIONS
# -------------------------
X = vectorizer.transform(df["text"])

probs = model.predict_proba(X)[:, 1]

# Slightly safer threshold
preds = (probs >= 0.55).astype(int)

df["prediction"] = preds
df["probability"] = probs


# -------------------------
# SPLIT GROUPS
# -------------------------
identity_df = df[df["group"] == "identity"]
non_identity_df = df[df["group"] == "non_identity"]


# -------------------------
# METRIC FUNCTION
# -------------------------
def compute_metrics(group_df, group_name):

    y_true = group_df["label"]
    y_pred = group_df["prediction"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0,1]
    ).ravel()

    fpr = fp / (fp + tn + 1e-10)

    fnr = fn / (fn + tp + 1e-10)

    tpr = tp / (tp + fn + 1e-10)

    accuracy = np.mean(y_true == y_pred)

    print(f"\n{group_name}")
    print("Accuracy:", round(accuracy, 4))
    print("FPR:", round(fpr, 4))
    print("FNR:", round(fnr, 4))
    print("Equalized Odds (TPR):", round(tpr, 4))

    return accuracy, fpr, fnr, tpr


# -------------------------
# IDENTITY GROUP
# -------------------------
identity_acc, identity_fpr, identity_fnr, identity_tpr = compute_metrics(
    identity_df,
    "IDENTITY GROUP"
)


# -------------------------
# NON-IDENTITY GROUP
# -------------------------
non_acc, non_fpr, non_fnr, non_tpr = compute_metrics(
    non_identity_df,
    "NON-IDENTITY GROUP"
)


# -------------------------
# DEMOGRAPHIC PARITY DIFFERENCE
# -------------------------
identity_positive_rate = np.mean(
    identity_df["prediction"]
)

non_identity_positive_rate = np.mean(
    non_identity_df["prediction"]
)

dp_difference = abs(
    identity_positive_rate - non_identity_positive_rate
)

print("\nDemographic Parity Difference:")
print(round(dp_difference, 4))


# -------------------------
# SUBGROUP AUC
# -------------------------
auc = roc_auc_score(
    identity_df["label"],
    identity_df["probability"]
)

print("\nSubgroup AUC:")
print(round(auc, 4))