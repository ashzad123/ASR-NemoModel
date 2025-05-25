# README.md

## FastAPI ASR Application (NVIDIA NeMo, ONNX)

### Overview
This project provides a containerized FastAPI server for automatic speech recognition (ASR) using an NVIDIA NeMo Hindi Conformer CTC model, optimized for ONNX inference.

---

### Setup Instructions

#### 0. Download the Model File

**Important:**  
Due to Git LFS storage limits, the `.nemo` model file is **not included in this repository**.

- **Download the model manually** from the official NVIDIA NGC link:  
  [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium/files](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium/files)
- **Place the downloaded `.nemo` file** (e.g., `stt_hi_conformer_ctc_medium.nemo`) in the `app/model/` directory.

*Reason: The model file is large and exceeded the Git LFS quota, so it must be downloaded separately.*

---

#### 1. Run Locally (without Docker)
1. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Python dependencies:**
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Export the nemo model to ONNX**
   ```sh
   python app/utils/load_model.py
   ```

4. **Run the FastAPI server:**
   ```sh
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   or (if you have the fastapi CLI):
   ```sh
   fastapi dev app/main.py
   ```

---

#### 2. Run with Docker

1. **Build the Docker image:**
   ```sh
   docker build -t fastapi-asr .
   ```

2. **Run the container:**
   ```sh
   docker run -p 8000:8000 fastapi-asr
   ```

---

### API Usage

#### Endpoint

- **POST** `/transcribe`
- Accepts: `.wav` audio file (16kHz, mono, 5–10 seconds)

#### Example cURL Request

```sh
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@debug_uploaded.wav"
```

#### Sample Response

```json
{
  "text": "▁केशव▁के▁घर▁में▁चार▁खिड़कियां▁हैं▁कई▁लोग▁कुमार▁को▁पसंद▁करते▁हैं"
}
```


#### Sample Output Image

You can find a sample output screenshot below (see the `sample_output.png` in the project folder):

![Sample Output](examples/sampel%20output.png)

---

---

### File Structure

```
First500Assignment/
│
├── app/
│   ├── main.py                # FastAPI app entrypoint
│   ├── routes/
│   │   └── transcribe.py      # FastAPI route for /transcribe
│   ├── services/
│   │   └── transcribe.py      # Audio preprocessing and inference logic
│   ├── utils/
│   │   └── load_model.py      # Script to export NeMo model to ONNX
│   └── model/
│       ├── stt_hi_conformer_ctc_medium.nemo   # Original NeMo model
│       ├── stt_hi_conformer_ctc_medium.onnx   # Exported ONNX model
│       └── ...                                # Model configs, vocab, etc.
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization instructions
├── README.md                  # This file
├── Description.md             # Features, issues, limitations
└── (Optional test audio, etc.)
```

---

### Design Considerations

- **Model Optimization:** The NeMo model is exported to ONNX for efficient inference.
- **Validation:** The API checks file type, sample rate, and duration.
- **Containerization:** Uses a slim Python base image and only necessary system dependencies.
- **Async:** The endpoint is async, but ONNXRuntime inference is synchronous (see Description.md for details).
- **Extensibility:** Easily swap models or add preprocessing as needed.

---

### Known Limitations

- Only supports 16kHz mono WAV files of 5–10 seconds.
- CTC decoding is basic; may be improved for better accuracy.
- ONNX inference is synchronous due to library constraints.

---
