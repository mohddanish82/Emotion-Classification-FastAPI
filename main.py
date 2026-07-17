from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(title="Emotion Classification API")

# Model Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model")

# Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Emotion Classes
classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Emotion API Running! 🔥", "docs": "/docs"}

@app.post("/predict")
def predict(input_data: TextInput):
    try:
        inputs = tokenizer(
            input_data.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "emotion": classes[predicted_class],
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                classes[i]: round(prob.item() * 100, 2) 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))