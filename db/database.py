import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from databases import Database
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env")

database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)
Base = declarative_base()
