from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load Dataset
dataset = load_dataset("dair-ai/emotion")

labels = dataset["train"].features["label"].names

# Model Name
model_name = "distilbert-base-uncased"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels)
)

model.to(device)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    predictions, labels_true = eval_pred
    predictions = predictions.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels_true, predictions),
        "f1": f1_score(labels_true, predictions, average="weighted")
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save Model
trainer.save_model("model")

# Save Tokenizer
tokenizer.save_pretrained("model")

print("Model and Tokenizer Saved Successfully!")