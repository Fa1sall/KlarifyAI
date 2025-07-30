from fastapi import FastAPI
from app.api.endpoints import router as klarify_router
from db.database import engine, Base

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="KlarifyAI",description="AI-powered Q&A from PDF documents",
    version="1.0.0")

app.include_router(klarify_router, prefix="/hackrx")

"""
from fastapi import FastAPI
from app.api.endpoints import router as klarify_router
from db.database import database, engine, Base
import db.models as db_models

# Create tables if not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="KlarifyAI",description="AI-powered Q&A from PDF documents",
    version="1.0.0")

@app.on_event("startup")
async def on_startup():
    await database.connect()

@app.on_event("shutdown")
async def on_shutdown():
    await database.disconnect()

app.include_router(klarify_router, prefix="/hackrx")
"""