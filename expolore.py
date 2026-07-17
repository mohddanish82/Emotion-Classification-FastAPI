
from huggingface_hub import list_datasets

print("Popular Text Classification Datasets:\n")

for ds in list_datasets(filter="text-classification", limit=20):
    print(ds.id)

from datasets import load_dataset
import pandas as pd


emotion = load_dataset("dair-ai/emotion")

print(" Dataset Loaded Successfully!")
print("Splits:", list(emotion.keys()))

print("\nTrain size     :", len(emotion['train']))
print("Validation size:", len(emotion['validation']))
print("Test size      :", len(emotion['test']))

print(emotion)
emotion.set_format(type='pandas')

df = emotion['train'][:]
print(df.head())