from fastapi import APIRouter
from app.api import auth, chat, product, profile, test

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(product.router)
api_router.include_router(profile.router)
api_router.include_router(test.router)

__all__ = ["api_router"]