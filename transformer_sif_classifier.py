import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "xlm-roberta-base"

DATASET_PATH = "dataset/sif_sanket_synthetic_safety_reports_v3.csv"

OUTPUT_DIR = "./models/xlm_roberta_sif_v3"


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Total reports:", len(df))

X = df["report_text"].astype(str)
y = df["sif_potential"].astype(str)


# =========================================================
# ENCODE LABELS
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nLabel mapping:")
for label, encoded in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(label, "->", encoded)


# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================

# First: 80% temporary train, 20% final test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)


# From remaining 85%, take validation
# 15% of total = approximately 17.65% of remaining
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=(15 / 85),
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Final Test :", len(X_test))


# =========================================================
# TOKENIZER
# =========================================================

print("\nLoading XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# TOKENIZATION
# =========================================================

train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=256
)

val_encodings = tokenizer(
    list(X_val),
    truncation=True,
    padding=True,
    max_length=256
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=256
)


# =========================================================
# DATASET CLASS
# =========================================================

class SIFDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels


    def __getitem__(self, index):

        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


    def __len__(self):

        return len(self.labels)


# =========================================================
# CREATE DATASETS
# =========================================================

train_dataset = SIFDataset(
    train_encodings,
    list(y_train)
)

val_dataset = SIFDataset(
    val_encodings,
    list(y_val)
)

test_dataset = SIFDataset(
    test_encodings,
    list(y_test)
)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading XLM-R model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NO",
        1: "YES"
    },
    label2id={
        "NO": 0,
        "YES": 1
    }
)


# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    learning_rate=2e-5,

    weight_decay=0.01,

    logging_steps=25,

    save_strategy="epoch",

    eval_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    report_to="none"
)


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset
)


# =========================================================
# TRAIN
# =========================================================

print("\n========================================")
print("STARTING XLM-R TRAINING")
print("========================================")

trainer.train()


# =========================================================
# VALIDATION EVALUATION
# =========================================================

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)

print(validation_results)


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)

print(test_results)


# =========================================================
# SAVE FINAL MODEL
# =========================================================

print("\nSaving model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("Model saved at:")
print(OUTPUT_DIR)
