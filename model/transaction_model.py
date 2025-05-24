from sqlalchemy import Column, Integer, String, ForeignKey, Numeric, DateTime, Enum
from sqlalchemy.orm import relationship
from database import Base
import datetime, enum

class PaymentMethodEnum(enum.Enum):
    CARD = "CARD"
    POINTS = "POINTS"

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    vehicle_number = Column(String(32), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    paid_with = Column(Enum(PaymentMethodEnum), nullable=False)
    card_id = Column(Integer, ForeignKey("cards.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # user = relationship("User", back_populates="transactions")
    # card = relationship("Card", back_populates="transactions")