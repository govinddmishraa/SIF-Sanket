import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="SIF-Sanket API",
    description="AI-powered SIF precursor detection system",
    version="1.0.0"
)


# --------------------------------------------------
# Model Configuration
# --------------------------------------------------

MODEL_PATH = "./models/xlm_roberta_sif_v4"
MODEL_NAME = "xlm-roberta-base"


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading SIF-Sanket model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()

print(f"Model loaded successfully on {DEVICE}")


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class AnalyzeRequest(BaseModel):
    report_text: str


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "SIF-Sanket API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XLM-RoBERTa V4",
        "device": str(DEVICE)
    }


# --------------------------------------------------
# SIF Analysis Endpoint
# --------------------------------------------------

@app.post("/analyze")
def analyze_report(request: AnalyzeRequest):

    text = request.report_text.strip()

    if not text:
        return {
            "error": "Report text cannot be empty"
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_class = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_class].item()

    label_map = {
        0: "NO",
        1: "YES"
    }

    prediction = label_map[predicted_class]

    return {
        "report_text": text,
        "sif_prediction": prediction,
        "confidence": round(confidence, 4)
    }
