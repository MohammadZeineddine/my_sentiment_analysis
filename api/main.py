from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from transformers import pipeline
from sqlalchemy.orm import Session
from models import Review, Base, engine, SessionLocal
import threading

app = FastAPI(title="Sentiment Analysis API")


def load_model():
    global sentiment_pipeline
    print("Starting model loading.")
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")


@app.on_event("startup")
async def startup_event():
    load_model()
    Base.metadata.create_all(bind=engine)


class TextData(BaseModel):
    text: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/analyze")
async def analyze_sentiment(data: TextData, db: Session = Depends(get_db)):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    try:
        results = sentiment_pipeline(data.text)
        sentiment_result = results[0]['label']
        review = Review(text=data.text, sentiment=sentiment_result)
        db.add(review)
        db.commit()
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_db")
def test_db(db: Session = Depends(get_db)):
    try:
        # Attempt to fetch the first review (if any)
        first_review = db.query(Review).first()
        return {"first_review": first_review.text if first_review else "No reviews yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_status")
def model_status():
    return {"loaded": sentiment_pipeline is not None}


@app.get("/test_connection")
def test_connection():
    return {"message": "Connection successful"}
