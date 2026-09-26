import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "xlm-roberta-base"

DATASET_PATH = "dataset/sif_sanket_synthetic_safety_reports_v4.csv"

OUTPUT_DIR = "./models/xlm_roberta_sif_v4"

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("       LOADING V4 DATASET")
print("==========================================")

df = pd.read_csv(DATASET_PATH)

print("Total reports:", len(df))

X = df["report_text"].astype(str)
y_text = df["sif_potential"].astype(str)


# ============================================================
# LABEL MAPPING
# ============================================================

label_map = {
    "NO": 0,
    "YES": 1
}

y = y_text.map(label_map).values

print("\nLabel mapping:")
print("NO  -> 0")
print("YES -> 1")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n==========================================")
print("       CREATING DATA SPLITS")
print("==========================================")

# 15% final test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=SEED,
    stratify=y
)

# 15% validation from total dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=(15 / 85),
    random_state=SEED,
    stratify=y_temp
)

print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Final Test :", len(X_test))


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZATION
# ============================================================

print("Tokenizing data...")

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


# ============================================================
# DATASET CLASS
# ============================================================

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


train_dataset = SIFDataset(
    train_encodings,
    y_train
)

val_dataset = SIFDataset(
    val_encodings,
    y_val
)

test_dataset = SIFDataset(
    test_encodings,
    y_test
)


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    learning_rate=2e-5,

    weight_decay=0.01,

    logging_steps=50,

    save_strategy="epoch",

    eval_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_f1",

    greater_is_better=True,

    report_to="none",

    seed=SEED
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("\n==========================================")
print("       STARTING FINAL V4 TRAINING")
print("==========================================")

trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\n==========================================")
print("       VALIDATION RESULTS")
print("==========================================")

validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)

print(
    f"Accuracy : {validation_results['eval_accuracy']:.4f}"
)

print(
    f"Precision: {validation_results['eval_precision']:.4f}"
)

print(
    f"Recall   : {validation_results['eval_recall']:.4f}"
)

print(
    f"F1       : {validation_results['eval_f1']:.4f}"
)


# ============================================================
# FINAL HELD-OUT TEST
# ============================================================

print("\n==========================================")
print("       FINAL HELD-OUT TEST")
print("==========================================")

test_output = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    test_output.predictions,
    axis=1
)

test_labels = np.array(y_test)

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="binary",
    zero_division=0
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)


# ============================================================
# FINAL METRICS
# ============================================================

print("\n------------------------------------------")
print("FINAL TEST METRICS")
print("------------------------------------------")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["NO", "YES"],
        digits=4
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n==========================================")
print("       SAVING FINAL V4 MODEL")
print("==========================================")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nModel saved at:")
print(OUTPUT_DIR)

print("\n==========================================")
print("       FINAL V4 TRAINING COMPLETE")
print("==========================================")
