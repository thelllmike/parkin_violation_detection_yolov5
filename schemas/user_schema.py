from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr
    license_plate: str

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int

    class Config:
        orm_mode = True