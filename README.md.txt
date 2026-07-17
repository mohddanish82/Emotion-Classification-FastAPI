# Emotion Classification API

A FastAPI-based Emotion Classification API built using Hugging Face Transformers and PyTorch. The API predicts the emotion of a given text.

## Features

- Emotion classification using a Transformer model
- FastAPI REST API
- Interactive Swagger UI
- JSON request and response
- Easy to deploy and extend

## Tech Stack

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- Scikit-learn

## Project Structure

```
Emotion-Classification-FastAPI/
│
├── model/
├── main.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/your-username/Emotion-Classification-FastAPI.git

cd Emotion-Classification-FastAPI

pip install -r requirements.txt
```

## Run the API

```bash
uvicorn main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000/docs
```

## Example Request

```json
{
  "text": "I am very happy today!"
}
```

## Example Response

```json
{
  "emotion": "joy",
  "confidence": 0.998
}
```

## Future Improvements

- Deploy on Render or Railway
- Docker support
- Batch prediction
- Frontend integration

## Author

Mohd Danish