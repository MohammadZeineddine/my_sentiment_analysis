import re
from langdetect import detect, LangDetectException
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from pydantic import BaseModel
from transformers import pipeline
from sqlalchemy.orm import Session
from sqlalchemy import func, case, cast, Date
from .models import Review, Base, engine, SessionLocal
from datetime import datetime
import logging

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")


def load_model():
    """
    Loads the sentiment analysis model.

    This function initializes the global variable `sentiment_pipeline` with the sentiment analysis model.
    The model used is "distilbert-base-uncased-finetuned-sst-2-english".

    Raises:
        Exception: If there is an error while loading the model.

    Returns:
        None
    """
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
    """
    Function to be executed on application startup.
    It loads the model and creates the necessary database tables.
    """
    load_model()
    Base.metadata.create_all(bind=engine)


@app.get("/")
async def home(request: Request):
    """
    This function handles the home route of the API.

    Parameters:
    - request: The incoming request object.

    Returns:
    - A TemplateResponse object with the "home.html" template and the request object.
    """
    return templates.TemplateResponse("home.html", {"request": request})


class TextData(BaseModel):
    """
    Represents a piece of text data.

    Attributes:
        text (str): The text data.
    """
    text: str


def get_db():
    """
    Returns a database session.

    Yields:
        db: The database session.

    Finally:
        Closes the database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_sentiment(request: Request, text: str = Form(...), db: Session = Depends(get_db)):
    """
    Analyzes the sentiment of a given text.

    Args:
        request (Request): The HTTP request object.
        text (str): The text to be analyzed.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the sentiment analysis result.

    Raises:
        HTTPException: If the text is empty, too long, or contains invalid characters.
        HTTPException: If the language is not supported for analysis.
        HTTPException: If the sentiment model is not loaded.
        HTTPException: If an error occurs during sentiment analysis.
    """
    if not text or len(text) > 500 or len(text) < 10:
        raise HTTPException(
            status_code=400, detail="Text must be between 10 and 500 characters.")

    if not re.match("^[a-zA-Z0-9 .,!?']+$", text):
        raise HTTPException(
            status_code=400, detail="Text contains invalid characters.")

    language = detect(text)
    if language not in ['en']:
        return {"error": "Unsupported language for analysis"}

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
                         "confidence": f"{sentiment_score*100:.2f}"}
        return templates.TemplateResponse("result.html", {"request": request, "result": template_data})
    except LangDetectException:
        return {"error": "Language detection failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reviews")
async def get_reviews(request: Request, db: Session = Depends(get_db)):
    """
    Retrieve all reviews from the database.

    Parameters:
    - request: The incoming request object.
    - db: The database session.

    Returns:
    - A template response containing the "reviews.html" template and the retrieved reviews.

    Raises:
    - HTTPException: If there is an error retrieving the reviews from the database.
    """
    try:
        reviews = db.query(Review).all()
        return templates.TemplateResponse("reviews.html", {"request": request, "reviews": reviews})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

logger = logging.getLogger(__name__)


def get_sentiment_over_time(db: Session):
    """
    Retrieves the sentiment data over time from the database.

    Args:
        db (Session): The database session object.

    Returns:
        list: A list of dictionaries containing the sentiment data over time.
            Each dictionary contains the following keys:
            - 'review_date': The date of the review.
            - 'positive_reviews': The number of positive reviews on that date.
            - 'negative_reviews': The number of negative reviews on that date.

    Raises:
        HTTPException: If an error occurs while fetching the data from the database.
    """
    try:
        result = db.query(
            func.date(Review.created_at).label('review_date'),
            func.sum(case((Review.sentiment == 'POSITIVE', 1), else_=0)).label(
                'positive_reviews'),
            func.sum(case((Review.sentiment == 'NEGATIVE', 1), else_=0)).label(
                'negative_reviews')
        ).group_by(func.date(Review.created_at)).order_by('review_date').all()

        return [{'review_date': str(row.review_date), 'positive_reviews': row.positive_reviews, 'negative_reviews': row.negative_reviews} for row in result]
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred while fetching data: {str(e)}")


@app.get("/review_sentiments")
async def get_review_sentiments(db: Session = Depends(get_db)):
    """
    Retrieve review sentiments over time.

    Parameters:
    - db: The database session.

    Returns:
    - sentiments: A list of review sentiments over time.
    """
    sentiments = get_sentiment_over_time(db)
    return sentiments


@app.get("/graphical_reviews")
async def graphical_reviews(request: Request):
    """
    Endpoint for rendering the graphical_reviews.html template.

    Parameters:
    - request: The incoming request object.

    Returns:
    - A TemplateResponse object containing the rendered graphical_reviews.html template.
    """
    return templates.TemplateResponse("graphical_reviews.html", {"request": request})


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
