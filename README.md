# Aadhaar Validation Service (Phase 1 & Phase 2)

A high-performance local Aadhaar image parsing and OCR extraction service built with **FastAPI**, **PaddleOCR**, **OpenCV**, **PyMuPDF**, and **dependency-injector**.

---

## Capabilities

- **Multi-Format Input**: Accepts front and back images (`.jpg`, `.jpeg`, `.png`) or multi-page PDF e-Aadhaar documents.
- **In-Memory Processing**: Uploaded documents are parsed and processed in memory without permanently persisting original files to disk.
- **In-Project Model Cache**: Models (`PP-OCRv5_mobile_det`, `devanagari_PP-OCRv5_mobile_rec`) are stored inside the project `models/official_models/` directory.
- **Multi-Core & Concurrent Pipeline**:
  - Utilizes multi-worker engine pool (`ocr_pool_size=2`) with multi-threaded oneDNN / Intel MKL BLAS execution.
  - Page 1 and Page 2 are processed concurrently in parallel via `asyncio.gather`.
  - Full 2-page e-Aadhaar PDF turnaround in **~4.2 seconds** (15.5x faster than baseline 69s).
- **Aadhaar ROI Isolation**: Automatically detects and crops the Aadhaar card cut-out section ($y \ge 0.55 \times \text{height}$) from full-page documents.
- **PDF-to-Image Rendering**: Uses PyMuPDF at configurable DPI (`150` DPI default) with page and dimension limits.
- **Image Quality Analysis**:
  - Blur estimation using Laplacian variance
  - Brightness calculation
  - Resolution and dimension checks with warning flags
- **PaddleOCR 3.x Extraction**:
  - Devanagari (Hindi) and English text recognition (`hi` model)
  - Text lines, recognition confidence scores, and bounding-box coordinates
  - Thread-safe engine worker pool
- **Data & Field Extraction**:
  - **Aadhaar Number**: Supports 12-digit unmasked and masked (`XXXX XXXX 1234`) formats.
  - **Verhoeff Algorithm Checksum**: Verifies whether the 12th digit matches UIDAI checksum specifications.
  - **DOB / Year of Birth**: Extracts date of birth or year of birth.
  - **Gender**: Detects Male/Female/Transgender (both English and Hindi).
  - **Address & PIN Code**: Extracts address lines and 6-digit PIN codes.
  - **Document Classification**: Identifies document type as `AADHAAR_FRONT`, `AADHAAR_BACK`, `E_AADHAAR`, or `NOT_CLASSIFIED`.
- **Dependency Injection Framework**: Built with `dependency-injector` for seamless swapping and testing.

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app & lifespan service setup
│   ├── api/
│   │   ├── __init__.py
│   │   └── document_routes.py   # /api/v1/documents/images & /pdf endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic BaseSettings configuration
│   │   ├── container.py         # Dependency Injection IoC container
│   │   └── interfaces.py        # Protocol interfaces for loose coupling
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── document.py          # Pydantic schemas for quality, OCR & extracted data
│   └── services/
│       ├── __init__.py
│       ├── extractor_service.py # Aadhaar regex & Verhoeff validation
│       ├── file_service.py      # Upload validation, decoding, PDF conversion
│       ├── image_service.py     # Quality analysis, normalization & ROI isolation
│       ├── ocr_service.py       # Engine pool & multi-core PaddleOCR worker
│       └── processing_service.py # Concurrent multi-page pipeline orchestrator
├── models/
│   └── official_models/         # Pre-provisioned local PaddleOCR deep-learning models
├── tests/
│   ├── test_api.py              # API integration & DI swapping tests
│   └── test_extraction.py       # Verhoeff, regex, & unit tests
├── .env                         # Service environment variables
├── requirements.txt             # Project dependencies
└── README.md
```

---

## Setup & Running

### 1. Environment Setup (Python 3.11)

```powershell
# Using uv (recommended on Windows):
uv venv .venv --python 3.11
.venv\Scripts\activate

# Install PaddlePaddle CPU:
uv pip install paddlepaddle==3.2.0 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ --python .venv

# Install requirements:
uv pip install -r requirements.txt --python .venv
```

### 2. Start Service

```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Endpoints & Swagger Testing

When the server is running, navigate to:
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI Schema**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

> [!TIP]
> Navigating directly to `http://127.0.0.1:8000/` automatically redirects to `/docs`.

### Swagger Testing Enhancements
1. **Try It Out Enabled**: "Try it out" is enabled by default so you can immediately execute requests.
2. **Instant 1-Click Test**: Execute `POST /api/v1/documents/test-synthetic` directly in Swagger UI to test OCR and Verhoeff validation without needing any local file.
3. **Sample Test Card Download**: Use `GET /api/v1/documents/sample-test-image` in Swagger to download a sample card image for manual upload testing.

---

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "service": "Aadhaar Validation Service",
  "ocr_lang": "hi",
  "ocr_device": "cpu"
}
```

### 2. Upload Images (Front & Optional Back)
`POST /api/v1/documents/images`
- `front_image` (multipart/form-data, required): Front Aadhaar card image.
- `back_image` (multipart/form-data, optional): Back Aadhaar card image.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/images" `
  -F "front_image=@sample_front.jpg" `
  -F "back_image=@sample_back.jpg"
```

### 3. Upload PDF
`POST /api/v1/documents/pdf`
- `file` (multipart/form-data, required): PDF document (unlocked).

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/pdf" `
  -F "file=@eaadhaar.pdf"
```

---

## Sample Response Structure

```json
{
  "processing_status": "SUCCESS",
  "document_type": "E_AADHAAR",
  "authenticity_status": "NOT_VERIFIED",
  "processing_time_seconds": 4.251,
  "pages": [
    {
      "page_number": 1,
      "image_quality": {
        "width": 1755,
        "height": 1240,
        "blur_score": 142.3,
        "brightness": 158.4,
        "status": "GOOD",
        "warnings": []
      },
      "ocr": {
        "status": "SUCCESS",
        "full_text": "Government of India Sahil Sahebrao Mankar DOB: 17/11/2001 Male 5134 5777 0655",
        "lines": [
          {
            "text": "Government of India",
            "confidence": 0.985,
            "bbox": [487, 82, 1043, 150]
          },
          {
            "text": "Sahil Sahebrao Mankar",
            "confidence": 0.962,
            "bbox": [594, 310, 1180, 370]
          },
          {
            "text": "DOB: 17/11/2001",
            "confidence": 0.978,
            "bbox": [599, 392, 890, 437]
          },
          {
            "text": "Male",
            "confidence": 0.991,
            "bbox": [596, 467, 690, 514]
          },
          {
            "text": "5134 5777 0655",
            "confidence": 0.994,
            "bbox": [509, 877, 1058, 973]
          }
        ]
      },
      "extracted_data": {
        "aadhaar_number": "5134 5777 0655",
        "is_aadhaar_number_valid": true,
        "is_masked": false,
        "name": "Sahil Sahebrao Mankar",
        "dob": "17/11/2001",
        "gender": "MALE",
        "address": null,
        "pin_code": null,
        "document_side": "FRONT"
      }
    }
  ],
  "aggregated_data": {
    "aadhaar_number": "5134 5777 0655",
    "is_aadhaar_number_valid": true,
    "is_masked": false,
    "name": "Sahil Sahebrao Mankar",
    "dob": "17/11/2001",
    "gender": "MALE",
    "address": "Address: S/O Sahebrao Mankar, gavthan, taluka- ambegaon, Pargaon Tarf Khed,, Pune, Maharashtra, 410512",
    "pin_code": "410512",
    "document_side": "FULL_PAGE"
  },
  "message": "Image processing and OCR completed successfully."
}
```
