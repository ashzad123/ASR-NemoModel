from fastapi import APIRouter,status,File, UploadFile
from app.services.transcribe import transcribe_audio

router = APIRouter()

@router.post("",description="Transcribe audio file",status_code=status.HTTP_201_CREATED)
async def upload_audio_file(file: UploadFile = File(...)):
    return await transcribe_audio(file)
