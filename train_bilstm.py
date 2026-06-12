import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

print("BiLSTM TRAINING STARTED")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("final_combined_dataset.csv")

# Reduce size for speed
df = df.sample(5000, random_state=42)

print("Dataset shape:", df.shape)

texts = df["text"].astype(str)
labels = df["label"]

# -------------------------
# TOKENIZATION
# -------------------------
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)

y = np.array(labels)

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL
# -------------------------
model = Sequential()

model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# TRAIN
# -------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -------------------------
# EVALUATE
# -------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nBiLSTM RESULTS")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))