"""Authentication: register, login, profile."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models.user import User
from app.schemas.auth import Token, UserCreate, UserLogin, UserOut
from app.services import auth_service
from app.utils.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut)
async def register(data: UserCreate, db: Annotated[AsyncSession, Depends(get_db)]) -> User:
    existing = await auth_service.get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return await auth_service.create_user(db, data)


@router.post("/login", response_model=Token)
async def login(data: UserLogin, db: Annotated[AsyncSession, Depends(get_db)]) -> Token:
    user = await auth_service.authenticate(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=user.email, role=user.role)
    return Token(access_token=token)


@router.get("/me", response_model=UserOut)
async def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user
