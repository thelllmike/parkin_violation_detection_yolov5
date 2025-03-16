from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from database import Base
import datetime

class ParkingViolation(Base):
    __tablename__ = "parking_violations"

    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(255), index=True, nullable=False)
    violation_time = Column(DateTime, default=datetime.datetime.utcnow)
    fine_amount = Column(Float, nullable=False)
    paid = Column(Boolean, default=False)
    description = Column(String(255), nullable=True)