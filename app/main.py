from typing import Union
from app.routes import router
from fastapi import FastAPI

app = FastAPI(title="Audio Transcription API", version="1.0.0" ,docs_url="/api/v1/docs")
app.include_router(router)