"""
FastAPI entrypoint: async API, OpenAPI docs, CORS, ML warmup, structured logging.
Run locally: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` from `/backend`.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routes import advanced, auth, medical
from app.services.ml_service import ml_service

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medical_ai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup: directories, DB schema, model weights. Shutdown: nothing special."""
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    Path("data/heatmaps").mkdir(parents=True, exist_ok=True)
    Path("data/masks").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    await init_db()
    ml_service.load_models()

    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    if admin_email and admin_password:
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import AsyncSession

        from app.database import AsyncSessionLocal
        from app.models.user import User, UserRole
        from app.utils.security import hash_password

        async with AsyncSessionLocal() as session:  # type: AsyncSession
            r = await session.execute(select(User).where(User.email == admin_email))
            if r.scalar_one_or_none() is None:
                session.add(
                    User(
                        email=admin_email,
                        hashed_password=hash_password(admin_password),
                        full_name="System Admin",
                        role=UserRole.ADMIN,
                    )
                )
                await session.commit()
                logger.info("Seeded admin user %s", admin_email)

    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan, version="1.0.0")

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(medical.router, prefix=settings.api_prefix)
app.include_router(advanced.router, prefix=settings.api_prefix)


@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.app_name}


@app.get("/")
async def root():
    return {"message": "AI Medical Diagnosis API", "docs": "/docs", "openapi": "/openapi.json"}
