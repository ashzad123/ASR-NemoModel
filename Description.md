# Description.md

## Features Implemented

- ✅ FastAPI server with `/transcribe` endpoint for audio transcription.
- ✅ Input validation for file type, sample rate, and duration.
- ✅ Model export from NeMo to ONNX for optimized inference.
- ✅ ONNXRuntime-based inference pipeline.
- ✅ Documentation with setup and usage instructions.

---

## Issues Encountered

- **ONNX CTC Decoding:** ONNXRuntime does not provide built-in CTC decoding. Implemented a basic greedy decoder, but accuracy may be lower than beam search or language model-assisted decoding.
- **Async Inference:** ONNXRuntime inference is synchronous. True async inference would require a different backend or process pool.
- **Model Input Requirements:** The model expects 16kHz mono audio. Additional preprocessing is required for other formats.
- **Large Docker Image:** Including NeMo and audio dependencies increases image size.
- **Docker Build Failure Due to Storage:** The Docker build could not be completed because my local storage limit was reached. As a result, I was unable to successfully build and test the containerized deployment.

---

## Unimplemented/Partial Components

- **Advanced CTC Decoding:** Did not implement beam search or language model integration for decoding due to time constraints.
- **Streaming/Batch Inference:** Only single-file, non-streaming inference is supported.
- **Automated Testing/CI:** No automated tests or CI/CD pipeline included.
- **Containerization Not Fully Tested:** Due to storage constraints, the Docker container build and test could not be completed.

---

## How to Overcome Challenges

- **Improve Decoding:** Integrate a CTC beam search decoder or use external libraries for better transcription accuracy.
- **Async Inference:** Use a thread/process pool or switch to an async-compatible inference backend.
- **Support More Formats:** Add preprocessing to handle stereo audio and resample to 16kHz as needed.
- **Reduce Image Size:** Use multi-stage builds, a `.dockerignore` file, or strip unnecessary dependencies to reduce Docker image size.
- **Resolve Storage Issues:** Use a machine with more available disk space or leverage cloud-based build services for Docker containerization.

---

## Known Limitations & Assumptions

- Only Hindi language is supported (per model).
- Audio files must be 5–10 seconds, 16kHz, mono WAV.
- Inference is synchronous.
- No authentication or rate limiting on the API.
- Containerization was not fully validated due to local storage limitations.

---
