import re
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
from sqlalchemy.orm import Session
from .models import Review, Base, engine, SessionLocal
from datetime import datetime

app = FastAPI(title="Sentiment Analysis API")

templates = Jinja2Templates(directory="templates")


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


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


class TextData(BaseModel):
    text: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/analyze")
async def analyze_sentiment(request: Request, text: str = Form(...), db: Session = Depends(get_db)):
    # Validation: Non-empty and length check
    if not text or len(text) > 500 or len(text) < 10:
        raise HTTPException(
            status_code=400, detail="Text must be between 10 and 500 characters.")

    # Validation: Character check (optional, customize regex as needed)
    if not re.match("^[a-zA-Z0-9 .,!?']+$", text):
        raise HTTPException(
            status_code=400, detail="Text contains invalid characters.")

    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        results = sentiment_pipeline(text)
        sentiment_label = results[0]['label']
        sentiment_score = results[0]['score']

        review = Review(text=text, sentiment=sentiment_label,
                        created_at=datetime.utcnow())
        db.add(review)
        db.commit()

        template_data = {"sentiment": sentiment_label,
                         "confidence": f"{sentiment_score:.2f}"}
        return templates.TemplateResponse("result.html", {"request": request, "results": template_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# add a rout to get all the reviews
@app.get("/reviews")
async def get_reviews(request: Request, db: Session = Depends(get_db)):
    try:
        reviews = db.query(Review).all()
        return templates.TemplateResponse("reviews.html", {"request": request, "reviews": reviews})
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
