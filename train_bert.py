import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments


print("BERT TRAINING STARTED")


# -------------------------
# LOAD DATA
# -------------------------
jigsaw = pd.read_csv("jigsaw_cleaned.csv")
hate = pd.read_csv("hate_cleaned.csv")
civil = pd.read_csv("civil_cleaned.csv")

# handle missing values
for df in [jigsaw, hate, civil]:
    df['text'] = df['text'].fillna("")

# CPU-friendly dataset size
jigsaw = jigsaw.sample(3000, random_state=42)
hate = hate.sample(3000, random_state=42)
civil = civil.sample(3000, random_state=42)

df = pd.concat([jigsaw, hate, civil])

print("Dataset size:", df.shape)


# -------------------------
# SPLIT
# -------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)


# -------------------------
# TOKENIZER
# -------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)


# -------------------------
# DATASET CLASS
# -------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)


# -------------------------
# MODEL
# -------------------------
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)


# -------------------------
# METRICS
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# -------------------------
# TRAINING SETTINGS (FIXED VERSION)
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,              # guide requirement
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# -------------------------
# TRAIN
# -------------------------
print("Training BERT...")
trainer.train()


# -------------------------
# EVALUATE
# -------------------------
results = trainer.evaluate()

print("\nFINAL BERT RESULTS")
print("Accuracy:", results["eval_accuracy"])
print("Precision:", results["eval_precision"])
print("Recall:", results["eval_recall"])
print("F1 Score:", results["eval_f1"])