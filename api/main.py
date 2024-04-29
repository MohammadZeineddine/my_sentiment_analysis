from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis API")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


class TextData(BaseModel):
    text: str


@app.post("/analyze")
async def analyze_sentiment(data: TextData):
    try:
        # Use the pipeline to analyze sentiment
        results = sentiment_pipeline(data.text)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
