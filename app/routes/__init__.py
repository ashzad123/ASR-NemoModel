from .transcribe import router as transcribe_router

from fastapi import APIRouter

router = APIRouter()

router.include_router(transcribe_router, prefix="/transcribe", tags=["Transcribe Audio"])