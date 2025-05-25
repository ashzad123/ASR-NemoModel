from pydantic import BaseModel

class TranscribeResponseSchema(BaseModel):
    """
    Schema for the response of the transcribe endpoint.
    """
    text: str | None = None