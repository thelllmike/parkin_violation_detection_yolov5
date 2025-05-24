# models/cards_model.py

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime

class Card(Base):
    __tablename__ = "cards"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    card_token  = Column(String(255), nullable=False)    # ← give it a max length
    last4       = Column(String(4), nullable=False)
    exp_month   = Column(Integer, nullable=False)
    exp_year    = Column(Integer, nullable=False)
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
