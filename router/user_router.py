from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from crud.user_crud import create_user, get_user_by_email, pwd_context
from schemas.user_schema import UserCreate, UserOut
from database import get_db, engine
from model.users_model import Base

# Create the database tables if they don't exist
Base.metadata.create_all(bind=engine)

router = APIRouter()

@router.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db, user)

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_email(db, email=form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    # For demonstration, a dummy token is returned. In production, use JWT or similar.
    return {"access_token": user.email, "token_type": "bearer"}