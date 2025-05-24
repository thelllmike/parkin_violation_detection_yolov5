# router/payment_router.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from crud.payment_crud import (
    CardCreate,
    create_card,
    get_cards,
    topup_points,
    process_card_payment,
    process_point_payment,
)
from schemas.payment_schemas import (
    CardCreate,
    CardOut,
    TopUpIn,
    PointsOut,
    PaymentIn,
    PaymentOut,
)

router = APIRouter(prefix="/users", tags=["payments"])

@router.post("/{user_id}/cards", response_model=CardOut)
def add_card(user_id: int, payload: CardCreate, db: Session = Depends(get_db)):
    # call your CRUD function, which returns the ORM Card instance
    card = create_card(
        db,
        user_id,
        payload.card_token,
        payload.last4,
        payload.exp_month,
        payload.exp_year,
    )
    return card  # FastAPI will serialize this to CardOut

@router.get("/{user_id}/cards", response_model=List[CardOut])
def list_cards(user_id: int, db: Session = Depends(get_db)):
    return get_cards(db, user_id)

@router.post("/{user_id}/topup/points", response_model=PointsOut)
def topup_points_endpoint(user_id: int, payload: TopUpIn, db: Session = Depends(get_db)):
    lp = topup_points(db, user_id, payload.amount)
    return {"points_balance": lp.points_balance}

@router.post("/{user_id}/pay/card", response_model=PaymentOut)
def pay_via_card(
    user_id: int,
    payload: PaymentIn,
    db: Session = Depends(get_db),
):
    if payload.amount is None or payload.card_id is None:
        raise HTTPException(
            status_code=400,
            detail="Both `amount` and `card_id` are required for card payments",
        )
    try:
        return process_card_payment(
            db,
            user_id,
            payload.vehicle_number,
            payload.amount,
            payload.card_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{user_id}/pay/points", response_model=PaymentOut)
def pay_via_points(
    user_id: int,
    payload: PaymentIn,
    db: Session = Depends(get_db),
):
    if payload.amount is None:
        raise HTTPException(
            status_code=400,
            detail="`amount` is required for points payments",
        )
    try:
        return process_point_payment(
            db,
            user_id,
            payload.vehicle_number,
            payload.amount,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))