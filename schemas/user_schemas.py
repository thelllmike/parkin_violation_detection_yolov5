# schemas/user_schemas.py

from pydantic import BaseModel, constr
from datetime import datetime
from typing import Optional, List

class UserAppCreate(BaseModel):
    eth_address: constr(min_length=42, max_length=42)
    private_key: str

class UserAppOut(BaseModel):
    id: int
    eth_address: str
    created_at: datetime

    class Config:
        orm_mode = True

# Aliases so you can import UserCreate/UserOut as expected
UserCreate = UserAppCreate
UserOut    = UserAppOut