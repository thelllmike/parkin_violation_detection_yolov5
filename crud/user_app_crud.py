# crud/user_app_crud.py

from sqlalchemy.orm import Session
from model.user_model import User
from schemas.user_schemas import UserAppCreate
from utils.security import encrypt_key
from typing import Optional

def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).get(user_id)

def get_user_by_address(db: Session, eth_address: str) -> Optional[User]:
    return db.query(User).filter(User.eth_address == eth_address).first()

def get_all_users(db: Session) -> list[User]:
    return db.query(User).all()

def create_user(db: Session, user_in: UserAppCreate) -> User:
    encrypted = encrypt_key(user_in.private_key)
    db_user = User(
        eth_address=user_in.eth_address,
        private_key_encrypted=encrypted
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user