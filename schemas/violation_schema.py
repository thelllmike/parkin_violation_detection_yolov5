from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ViolationBase(BaseModel):
    license_plate: str
    fine_amount: float
    description: Optional[str] = None

class ViolationCreate(ViolationBase):
    pass

class ViolationOut(ViolationBase):
    id: int
    violation_time: datetime
    paid: bool

    class Config:
        orm_mode = True