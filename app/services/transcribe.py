from fastapi import HTTPException, File
import wave
import contextlib
from loguru import logger
import onnxruntime as ort
import numpy as np
import io
import soundfile as sf
import librosa

# Load ONNX model once at module level
onnx_session = ort.InferenceSession("app/model/stt_hi_conformer_ctc_medium.onnx")

async def transcribe_audio(file: File):
    """
    Transcribe audio file using ONNX model.
    """
    # Read audio bytes
    audio_bytes = await file.read()
    # Validate the audio file using bytes
    validate_audio_bytes(audio_bytes)
    debug_audio_path = "input_file.wav"
    with open(debug_audio_path, "wb") as f:
        f.write(audio_bytes)
    # Decode audio
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    features = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=80,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window='hann',
        fmin=0,
        fmax=8000,
        power=2.0
    )
    log_mel = np.log(features + 1e-6)
    # Per-feature normalization
    mean = np.mean(log_mel, axis=1, keepdims=True)
    std = np.std(log_mel, axis=1, keepdims=True) + 1e-9
    log_mel = (log_mel - mean) / std
    log_mel = np.expand_dims(log_mel, 0)
    length = np.array([log_mel.shape[2]], dtype=np.int64)  # shape (1,)

    ort_inputs = {
        "audio_signal": log_mel,
        "length": length
    }
    ort_outs = onnx_session.run(None, ort_inputs)
    return {
        "text": decode_output(np.array(ort_outs[0])),
    }

def validate_audio_bytes(audio_bytes: bytes):
    """
    Validate the audio file from bytes.
    """
    try:
        with contextlib.closing(wave.open(io.BytesIO(audio_bytes), 'rb')) as f:
            if f.getcomptype() != 'NONE':
                raise HTTPException(status_code=400, detail="File is not in WAV format.")
            if f.getnchannels() != 1:
                raise HTTPException(status_code=400, detail="File is not mono channel.")
            if f.getframerate() != 16000:
                logger.error(f"Sample rate is {f.getframerate()}Hz, expected 16000Hz.")
                raise HTTPException(status_code=400, detail="Sample rate is not 16kHz.")
    except HTTPException as e:
        raise e
    except wave.Error as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail="Invalid audio file")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail="An error occurred while validating the audio file")

def decode_output(output):
    """
    Greedy CTC decoder for model output (Hindi BPE vocab).
    Assumes output is (batch, time, vocab_size) or (time, vocab_size).
    Returns the decoded string for the first batch item.
    """
    # If output is 3D, take the first batch
    if len(output.shape) == 3:
        output = output[0]
    pred_ids = np.argmax(output, axis=-1)
    decoded = []
    prev = -1
    blank_id = 0  

    vocab = [
        '<unk>', 'ा', 'र', 'ी', '▁', 'े', 'न', 'ि', 'त', 'क', '्', 'ल', 'म', 'स', 'ं', '▁स', 'ह', 'ो', 'ु', 'द', 'य',
        'प', '▁है', '▁के', 'ग', '▁ब', '▁म', 'व', '▁क', '▁में', 'ट', '▁अ', 'ज', '▁द', '▁प', '▁आ', '्र', 'ू', '▁ज',
        '▁की', '▁र', 'ध', 'र्', 'ों', 'ख', '▁का', '्य', 'च', 'ए', 'ब', 'भ', 'ने', '▁को', '▁से', '▁ल', '▁और', '▁प्र',
        '▁त', '▁कर', '▁व', 'ता', 'श', '▁कि', '▁ह', '▁न', '▁ग', 'ना', '▁हो', 'ै', '▁पर', 'थ', '▁उ', 'ड', '▁च', 'िक',
        'णण', 'ई', '▁हैं', 'िया', '▁इस', 'फ', '▁वि', 'वा', '▁जा', 'ष', 'ित', '▁श', 'ें', '▁ने', 'ेश', 'ते', 'इ',
        '▁भी', 'का', '▁एक', '्या', '▁हम', '▁सं', 'िल', 'ंग', 'ड़', 'छ', 'क्ष', 'ौ', 'ठ', '़', 'ॉ', 'ओ', 'ढ', 'घ',
        'आ', 'झ', 'ऐ', 'ँ', 'ऊ', 'उ', 'ः', 'औ', ',', 'ऍ', 'ॅ', 'ॠ', 'ऋ', 'ऑ', 'ञ', 'ृ', 'अ', 'ङ'
    ]

    for idx in pred_ids:
        if idx != prev and idx != blank_id:
            if idx < len(vocab):
                decoded.append(vocab[idx])
        prev = idx
    return "".join(decoded) if decoded else ""