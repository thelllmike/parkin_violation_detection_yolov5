# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# # MySQL Database URL
# SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://root:12345678@localhost:3306/parkviolation"

# # Create the engine to connect to the MySQL database
# engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

# # Create a configured "Session" class
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Create a Base class for declarative models to inherit from
# Base = declarative_base()

# # Dependency to get DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# db.py
# db.py

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# ─── Load env vars ───────────────────────────────────────────────────────────────
load_dotenv()  # Make sure you have your .env with DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

DB_USER     = os.getenv("DB_USER", "mahi")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mahi@#Gmail.com")
DB_HOST     = os.getenv("DB_HOST", "mahi.mysql.database.azure.com")
DB_PORT     = os.getenv("DB_PORT", "3306")
DB_NAME     = os.getenv("DB_NAME", "parkviolation")

# URL-encode any special chars in the password
password_escaped = quote_plus(DB_PASSWORD)

# Build your connection URL
SQLALCHEMY_DATABASE_URL = (
    f"mysql+mysqlconnector://{DB_USER}:{password_escaped}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Create the engine (no connect_args here)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative base
Base = declarative_base()

# FastAPI dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()