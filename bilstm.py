import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

print("BiLSTM TRAINING STARTED")


# =====================================
# LOAD DATA
# =====================================

jigsaw = pd.read_csv("jigsaw_cleaned.csv")
hate = pd.read_csv("hate_cleaned.csv")
civil = pd.read_csv("civil_cleaned.csv")

for df in [jigsaw, hate, civil]:
    df["text"] = df["text"].fillna("")

# Larger dataset
jigsaw = jigsaw.sample(5000, random_state=42)
hate = hate.sample(5000, random_state=42)
civil = civil.sample(5000, random_state=42)

df = pd.concat([jigsaw, hate, civil])

print("Dataset shape:", df.shape)


# =====================================
# SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# =====================================
# TOKENIZATION
# =====================================

MAX_WORDS = 20000
MAX_LEN = 200

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)


# =====================================
# MODEL
# =====================================

model = Sequential()

model.add(
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    )
)

model.add(
    Bidirectional(
        LSTM(
            128,
            return_sequences=False
        )
    )
)

model.add(Dropout(0.5))

model.add(Dense(
    64,
    activation="relu"
))

model.add(Dropout(0.3))

model.add(Dense(
    1,
    activation="sigmoid"
))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())


# =====================================
# CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# =====================================
# TRAIN
# =====================================

history = model.fit(
    X_train_pad,
    y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)


# =====================================
# EVALUATE
# =====================================

pred_prob = model.predict(X_test_pad)

pred = (
    pred_prob > 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    pred
)

print("\nBiLSTM RESULTS")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred
    )
)