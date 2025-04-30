from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import torch
import numpy as np

# Use a model trained on informal chat data
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1

# Load pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

def normalize_label(label):
    """Convert raw model labels to standard form."""
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
        "NEG": "NEGATIVE", "NEU": "NEUTRAL", "POS": "POSITIVE"
    }
    return label_map.get(label.upper(), label.upper())

def analyze_sentiment(df, selected_user='Overall'):
    # Filter by user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Clean and filter messages
    df = df.copy()
    df['message'] = df['message'].astype(str)
    df = df[df['message'].str.strip().astype(bool)]  # Remove empty messages

    # Run inference in batches to avoid memory issues
    messages = df['message'].tolist()
    batch_size = 32
    results = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        try:
            outputs = sentiment_pipeline(batch, truncation=True)
            results.extend(outputs)
        except Exception as e:
            print(f"Batch failed: {e}")
            results.extend([{'label': 'NEUTRAL', 'score': 0.0}] * len(batch))  # fallback

    # Parse results
    df['sentiment'] = [normalize_label(r['label']) for r in results]
    df['score'] = [r['score'] for r in results]

    # Summary count
    sentiment_summary = df['sentiment'].value_counts().reset_index()
    sentiment_summary.columns = ['Sentiment', 'Count']

    return df, sentiment_summary
