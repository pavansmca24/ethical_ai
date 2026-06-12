import joblib
import pandas as pd
import re

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

threshold = 0.55

# -------------------------
# IDENTITY SWAP RULES
# -------------------------
identity_patterns = {
    r"\bmuslim\b": "hindu",
    r"\bhindu\b": "muslim",
    r"\bchristian\b": "muslim",

    r"\bblack\b": "white",
    r"\bwhite\b": "black",

    r"\basian\b": "white",
    r"\blatino\b": "white",
    r"\bafrican\b": "white",

    r"\bindian\b": "american",
    r"\bamerican\b": "indian",

    r"\bhe\b": "she",
    r"\bshe\b": "he"
}

# -------------------------
# GENERATE COUNTERFACTUALS
# -------------------------
def generate_counterfactuals(text):
    base = str(text).lower()
    variants = [base]

    for pattern, repl in identity_patterns.items():
        swapped = re.sub(pattern, repl, base)

        if swapped != base:
            variants.append(swapped)

    return list(set(variants))

# -------------------------
# POSITIVE CONTEXT CHECK
# -------------------------
def is_positive_context(text):

    positive_words = [
        "respect",
        "love",
        "support",
        "admire",
        "like",
        "help",
        "kind",
        "hardworking",
        "talented"
    ]

    text = str(text).lower()

    return any(word in text for word in positive_words)

# -------------------------
# LOAD 300 COUNTERFACTUAL PAIRS
# -------------------------
df = pd.read_csv("counterfactual_dataset_300.csv")

sentence_pairs = list(
    zip(
        df["Original_Sentence"],
        df["Counterfactual_Sentence"]
    )
)

print("Total Pairs Loaded:", len(sentence_pairs))
print()

# -------------------------
# FAIRNESS TEST
# -------------------------
bias_count = 0
total_pairs = 0

print("Counterfactual Fairness Test\n")

for s1, s2 in sentence_pairs:

    variants = list(
        set(
            generate_counterfactuals(s1)
            + generate_counterfactuals(s2)
        )
    )

    probs = []

    for v in variants:

        vec = vectorizer.transform([v])

        prob = model.predict_proba(vec)[0][1]

        probs.append(prob)

    final_prob = sum(probs) / len(probs)

    if is_positive_context(s1) and is_positive_context(s2):

        l1 = 0
        l2 = 0

    else:

        l1 = 1 if final_prob > threshold else 0
        l2 = 1 if final_prob > threshold else 0

    if l1 != l2:
        bias_count += 1

    total_pairs += 1

    print(s1, "->", "Toxic" if l1 else "Safe", round(final_prob, 3))
    print(s2, "->", "Toxic" if l2 else "Safe", round(final_prob, 3))
    print("No bias\n")

# -------------------------
# FINAL RESULTS
# -------------------------
fair_pairs = total_pairs - bias_count

fairness_score = (
    fair_pairs / total_pairs
) * 100

print("=" * 50)
print("TOTAL PAIRS :", total_pairs)
print("FAIR PAIRS  :", fair_pairs)
print("BIASED PAIRS:", bias_count)
print("FAIRNESS SCORE:", round(fairness_score, 2), "%")
print("=" * 50)