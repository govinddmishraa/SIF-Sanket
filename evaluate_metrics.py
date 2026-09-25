import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

print("==========================================")
print("     SIF-GUARD MODEL EVALUATION")
print("==========================================")

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "dataset/sif_sanket_synthetic_safety_reports_v2_1000.csv"
MODEL_PATH = "./models/xlm_roberta_sif/checkpoint-600"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ==========================================
# 2. LABEL MAPPING
# ==========================================

label_map = {
    "NO": 0,
    "YES": 1
}

df["label"] = df["sif_potential"].map(label_map)

print("\n========== LABEL DISTRIBUTION ==========")
print(df["sif_potential"].value_counts())

# ==========================================
# 3. SAME TRAIN-TEST SPLIT
# ==========================================

train_df = df.sample(
    frac=0.8,
    random_state=42
)

test_df = df.drop(train_df.index)

print("\n========== TEST DATA ==========")
print("Test samples:", len(test_df))

# ==========================================
# 4. LOAD TOKENIZER
# ==========================================

print("\n========== TOKENIZER ==========")

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)

# ==========================================
# 5. FIND TEXT COLUMN
# ==========================================

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

# Safety report text column
TEXT_COLUMN = "report_text"

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{TEXT_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# ==========================================
# 6. TOKENIZE TEST DATA
# ==========================================

def tokenize_function(examples):
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = Dataset.from_pandas(
    test_df[[TEXT_COLUMN, "label"]],
    preserve_index=False
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.remove_columns(
    [TEXT_COLUMN]
)

test_dataset.set_format("torch")

# ==========================================
# 7. LOAD TRAINED MODEL
# ==========================================

print("\n========== LOADING TRAINED MODEL ==========")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2
)

print("Model loaded successfully.")

# ==========================================
# 8. TRAINER
# ==========================================

trainer = Trainer(
    model=model
)

# ==========================================
# 9. PREDICTIONS
# ==========================================

print("\n========== GENERATING PREDICTIONS ==========")

predictions = trainer.predict(test_dataset)

logits = predictions.predictions

y_true = np.array(test_df["label"])

y_pred = np.argmax(logits, axis=1)

# ==========================================
# 10. METRICS
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

print("\n========== FINAL METRICS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["NO", "YES"],
        zero_division=0
    )
)

print("\n========== EVALUATION COMPLETE ==========")
