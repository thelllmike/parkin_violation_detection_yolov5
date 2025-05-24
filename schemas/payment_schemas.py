from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Literal

class CardCreate(BaseModel):
    card_token: str
    last4: str
    exp_month: int
    exp_year: int

class CardOut(CardCreate):
    id: int
    created_at: datetime
    class Config:
        orm_mode = True

class TopUpIn(BaseModel):
    amount: int

class PointsOut(BaseModel):
    points_balance: int

class PaymentIn(BaseModel):
    vehicle_number: str
    amount: Optional[float] = None
    card_id: Optional[int] = None

class PaymentOut(BaseModel):
    id: int
    user_id: int
    vehicle_number: str
    amount: float
    paid_with: Literal["CARD","POINTS"]
    created_at: datetime
    class Config:
        orm_mode = True