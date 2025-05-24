# model/users_model.py
from sqlalchemy import Column, Integer, String, DateTime, Text
from database import Base
import datetime

class User(Base):
    __tablename__ = "user"   # or "users", just be consistent

    id                     = Column(Integer, primary_key=True, index=True)
    eth_address            = Column(String(42), unique=True, index=True, nullable=False)
    private_key_encrypted  = Column(Text, nullable=True)
    created_at             = Column(DateTime, default=datetime.datetime.utcnow)

    # any relationships…