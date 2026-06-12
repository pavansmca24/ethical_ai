import joblib

print("Toxicity Prediction System")

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# -------------------------
# FUNCTION
# -------------------------
def predict_text(text):
    text_vec = vectorizer.transform([text])
    prob = model.predict_proba(text_vec)[0][1]

    # Higher threshold reduces false positives
    THRESHOLD = 0.80

    pred = 1 if prob >= THRESHOLD else 0

    return pred, prob

# -------------------------
# USER INPUT LOOP
# -------------------------
while True:
    user_input = input("\nEnter text (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    pred, prob = predict_text(user_input)

    if pred == 1:
        print(f"Toxic (confidence: {prob:.2f})")
    else:
        print(f"Non-Toxic (confidence: {1-prob:.2f})")