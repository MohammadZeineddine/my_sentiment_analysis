from sqlalchemy import Column, Integer, String, create_engine, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://user:password@db:5432/sentiment_analysis"
engine = create_engine(DATABASE_URL)
SessionLocal = scoped_session(sessionmaker(
    autocommit=False, autoflush=False, bind=engine))


Base = declarative_base()


class Review(Base):
    __tablename__ = 'reviews'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
