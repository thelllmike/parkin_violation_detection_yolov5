from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas.violation_schema import ViolationCreate, ViolationOut
from crud.violation_crud import create_violation, get_violation, get_all_violations, mark_violation_paid
from database import get_db
from typing import List

router = APIRouter()

@router.post("/", response_model=ViolationOut)
def add_violation(violation: ViolationCreate, db: Session = Depends(get_db)):
    return create_violation(db, violation)

@router.get("/{violation_id}", response_model=ViolationOut)
def read_violation(violation_id: int, db: Session = Depends(get_db)):
    db_violation = get_violation(db, violation_id)
    if db_violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return db_violation

@router.get("/", response_model=List[ViolationOut])
def read_violations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_all_violations(db, skip, limit)

@router.put("/{violation_id}/pay", response_model=ViolationOut)
def pay_violation(violation_id: int, db: Session = Depends(get_db)):
    db_violation = mark_violation_paid(db, violation_id)
    if not db_violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    return db_violation