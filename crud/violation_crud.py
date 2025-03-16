from sqlalchemy.orm import Session
from model.violation_model import ParkingViolation
from schemas.violation_schema import ViolationCreate

def create_violation(db: Session, violation: ViolationCreate):
    db_violation = ParkingViolation(**violation.dict())
    db.add(db_violation)
    db.commit()
    db.refresh(db_violation)
    return db_violation

def get_violation(db: Session, violation_id: int):
    return db.query(ParkingViolation).filter(ParkingViolation.id == violation_id).first()

def get_all_violations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(ParkingViolation).offset(skip).limit(limit).all()

def mark_violation_paid(db: Session, violation_id: int):
    db_violation = get_violation(db, violation_id)
    if db_violation:
        db_violation.paid = True
        db.commit()
        db.refresh(db_violation)
    return db_violation