from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from crud.user_app_crud import (
    get_user,
    get_user_by_address,
    get_all_users,
    create_user
)
from schemas.user_schemas import UserCreate, UserOut
from typing import List

router = APIRouter(prefix="/app/users", tags=["app_users"])

@router.post("/register", response_model=UserOut)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_address(db, user.eth_address):
        raise HTTPException(status_code=400, detail="Address already registered")
    return create_user(db, user)

@router.get("/{user_id}", response_model=UserOut)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return get_all_users(db)