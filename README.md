# Aadhaar Validation & Extraction Microservice

A privacy-focused, high-performance local Aadhaar image parsing, OCR extraction, and multi-factor authenticity validation service built with **FastAPI**, **PaddleOCR 3.x**, **OpenCV**, **PyMuPDF**, and **dependency-injector**.

---

## Capabilities

- **Multi-Format Input**: Accepts separate front/back images (`.jpg`, `.jpeg`, `.png`) or multi-page PDF e-Aadhaar documents.
- **In-Memory Privacy-First Processing**: Documents are processed strictly in volatile RAM without persisting unencrypted images, PDFs, or resident PII to disk.
- **In-Project Model Cache**: Deep learning models (`PP-OCRv5_mobile_det`, `devanagari_PP-OCRv5_mobile_rec`) are pre-cached inside the project repository at `models/official_models/`.
- **Three-Way Parallel Page Pipeline**:
  - Each page runs **Part 1 (OCR Extraction)**, **Part 2 (Format & Visual Validation)**, and **Part 3 (QR Extraction)** concurrently via `asyncio.gather`.
  - All document pages execute simultaneously across a multi-engine worker pool (`ocr_pool_size=2`) utilizing Intel oneDNN / MKL multi-threading.
  - Complete 2-page e-Aadhaar turnaround in **~3.8 seconds** (down from 68.8s baseline).
- **Two-Tier Side-Aware Authenticity Architecture**:
  - **Front Side**: Validates resident biometric portrait photo, Ashoka Pillar national emblem, official tricolor/maroon palette, and card geometry (ISO/IEC 7810 ID-1 CR80 ratio).
  - **Back Side**: Validates UIDAI 2D Secure QR Code, address block, and UIDAI contact stamps.
  - **Document Triangulation**: Cross-validates QR code payload against OCR-extracted demographics to detect Photoshop edits, swapped card sides, or fake documents.
- **UIDAI QR Code Decoding & Decompression**:
  - **Secure QR (V2/V3)**: Decompresses decimal BigInteger / Zlib binary payloads and parses Reference ID, Name, DOB, Gender, Care-Of, District, PIN code, State, and embedded photo presence.
  - **Legacy XML QR (V1)**: Decodes `<PrintLetterBarcodeData ... />` attributes.
- **Mathematical Validation**: Verifies 12-digit Aadhaar numbers against the standard UIDAI Verhoeff dihedral group $D_{10}$ checksum algorithm.
- **Card ROI Isolation**: Automatically isolates the identity card cut-out section ($y \ge 0.55 \times \text{height}$) from full-page A4 e-Aadhaar documents and camera scanner backgrounds.
- **Dependency Injection**: Built with `dependency-injector` for seamless component decoupling, swapping, and mocking in unit tests.

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, timing middleware, global error handling
│   ├── api/
│   │   ├── __init__.py
│   │   └── document_routes.py   # HTTP endpoints (/images, /pdf, /test-synthetic)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Streamlined Pydantic BaseSettings (9 operational knobs)
│   │   └── container.py         # IoC Dependency Injection Container (singleton lifecycles)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── document.py          # Pydantic schemas (Authenticity, Quality, OCR, Demographics)
│   └── services/
│       ├── __init__.py
│       ├── extractor_service.py # Regex extraction, Verhoeff checksum & classification
│       ├── media_service.py     # Buffer ingestion, PyMuPDF, OpenCV quality & visual checks
│       ├── ocr_service.py       # Engine pool & multi-core PaddleOCR worker
│       ├── processing_service.py# 3-way parallel orchestrator & holistic authenticity engine
│       └── qr_service.py        # Dual-format QR decoder (V2 Zlib & V1 XML) + Cross-match
├── models/
│   └── official_models/         # Pre-provisioned local PaddleOCR deep-learning models
├── tests/
│   ├── test_api.py              # API integration, DI swapping, and timing tests
│   └── test_extraction.py       # Verhoeff, regex, visual authenticity & QR tests
├── .env                         # Service environment variables
├── requirements.txt             # Project dependencies (including pyzbar)
└── README.md
```

---

## Setup & Running

### 1. Environment Setup (Python 3.11)

```powershell
# Create and activate virtual environment:
python -m venv .venv
.venv\Scripts\activate

# Install PaddlePaddle CPU:
pip install paddlepaddle==3.2.0 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install requirements:
pip install -r requirements.txt
```

### 2. Start Service

```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Endpoints

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "service": "Aadhaar Validation Service",
  "ocr_det_model": "PP-OCRv5_mobile_det",
  "ocr_rec_model": "devanagari_PP-OCRv5_mobile_rec",
  "ocr_pool_size": 2
}
```

### 2. Upload Images (Front & Optional Back)
`POST /api/v1/documents/images`
- `front_image` (multipart/form-data, required): Front Aadhaar card image (`.jpg`, `.png`).
- `back_image` (multipart/form-data, optional): Back Aadhaar card image (`.jpg`, `.png`).

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/images" `
  -F "front_image=@sample_front.jpg" `
  -F "back_image=@sample_back.jpg"
```

### 3. Upload Multi-Page PDF e-Aadhaar
`POST /api/v1/documents/pdf`
- `file` (multipart/form-data, required): PDF document (unlocked).

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/pdf" `
  -F "file=@eaadhaar.pdf"
```

### 4. Synthetic Swagger Test (1-Click Test)
`POST /api/v1/documents/test-synthetic`  
Instant test of OCR, layout classification, and Verhoeff validation without uploading a file.

---

## Response Structure Breakdown

The API returns a unified JSON response divided into 4 key sections:
1. **Metadata & Status**: Request status, overall document type, processing time.
2. **Holistic Authenticity (`authenticity`)**: Side-aware multi-factor security evaluation, tampering detection, and security flags.
3. **Aggregated Demographics (`aggregated_data`)**: Unified identity attributes merged across all pages.
4. **QR Cross-Match (`qr_cross_match`)**: Decoded QR payload and field-by-field comparison against OCR text.
5. **Per-Page Breakdown (`pages`)**: Granular quality diagnostics, page-level visual metrics, and raw OCR bounding boxes.

### Complete Response Example

```json
{
  "processing_status": "SUCCESS",
  "document_type": "E_AADHAAR",
  "authenticity_status": "VERIFIED_AUTHENTIC_CARD",
  "processing_time_seconds": 3.826,
  "message": "Image processing, OCR, and multi-factor authenticity verification completed successfully.",

  "authenticity": {
    "verdict": "AUTHENTIC_COMPLETE_PAIR",
    "confidence_score": 0.99,
    "completeness": "FULL_SET",
    "is_tampered": false,
    "has_front_face_or_portrait": true,
    "has_back_secure_qr": true,
    "has_valid_verhoeff_checksum": true,
    "has_official_color_palette": true,
    "has_valid_card_geometry": true,
    "qr_ocr_data_matched": true,
    "security_flags": [
      "PORTRAIT_BIOMETRIC_PRESENT",
      "SECURE_QR_PRESENT",
      "VERHOEFF_CHECKSUM_VALID",
      "OFFICIAL_PALETTE_DETECTED",
      "CARD_GEOMETRY_CR80_VALID",
      "QR_DEMOGRAPHICS_MATCHED",
      "DOCUMENT_FULL_SET_COMPLETE"
    ]
  },

  "aggregated_data": {
    "aadhaar_number": "5134 5777 0655",
    "is_aadhaar_number_valid": true,
    "is_masked": false,
    "name": "Sahil Sahebrao Mankar",
    "dob": "17/11/2001",
    "gender": "MALE",
    "address": "Address: S/O Sahebrao Mankar, gavthan, taluka- ambegaon, Pargaon Tarf Khed, Pune, Maharashtra, 410512",
    "pin_code": "410512",
    "document_side": "FULL_PAGE"
  },

  "qr_cross_match": {
    "cross_match_status": "MATCHED",
    "match_confidence": 1.0,
    "name_matched": true,
    "dob_matched": true,
    "gender_matched": true,
    "aadhaar_last4_matched": true,
    "pin_code_matched": true,
    "mismatches": [],
    "qr_data": {
      "format": "SECURE_QR_V2",
      "reference_id": "065520210713221106490",
      "aadhaar_last4": "0655",
      "name": "Sahil Sahebrao Mankar",
      "dob": "17/11/2001",
      "gender": "MALE",
      "care_of": "S/O Sahebrao Mankar",
      "pin_code": "410512",
      "district": "Pune",
      "state": "Maharashtra",
      "address": "S/O Sahebrao Mankar, gavthan  taluka- ambegaon, Pargaon Tarf Khed, Pune, Maharashtra, 410512",
      "has_photo": true
    }
  },

  "pages": [
    {
      "page_number": 1,
      "image_quality": {
        "width": 2105,
        "height": 1488,
        "blur_score": 142.3,
        "brightness": 158.4,
        "status": "GOOD",
        "warnings": []
      },
      "visual_authenticity": {
        "is_valid_aadhaar_card_image": true,
        "authenticity_score": 80,
        "aspect_ratio": 1.415,
        "qr_detected": false,
        "has_aadhaar_palette": true,
        "has_photo_or_graphics": true,
        "checks_passed": [
          "VALID_ASPECT_RATIO",
          "AADHAAR_PALETTE_DETECTED",
          "GRAPHICS_OR_PORTRAIT_REGION_DETECTED"
        ]
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
  ]
}
```

---

### Front-Only Upload Response Example

When only the front side is uploaded (`POST /api/v1/documents/images` with only `front_image`):

```json
{
  "processing_status": "SUCCESS",
  "document_type": "AADHAAR_FRONT",
  "authenticity_status": "VERIFIED_CARD_IMAGE",
  "processing_time_seconds": 1.45,
  "message": "Image processing, OCR, and multi-factor authenticity verification completed successfully.",
  "authenticity": {
    "verdict": "AUTHENTIC_FRONT_ONLY",
    "confidence_score": 0.90,
    "completeness": "FRONT_ONLY",
    "is_tampered": false,
    "has_front_face_or_portrait": true,
    "has_back_secure_qr": false,
    "has_valid_verhoeff_checksum": true,
    "has_official_color_palette": true,
    "has_valid_card_geometry": true,
    "qr_ocr_data_matched": null,
    "security_flags": [
      "PORTRAIT_BIOMETRIC_PRESENT",
      "VERHOEFF_CHECKSUM_VALID",
      "OFFICIAL_PALETTE_DETECTED",
      "CARD_GEOMETRY_CR80_VALID"
    ]
  },
  "aggregated_data": {
    "aadhaar_number": "5134 5777 0655",
    "is_aadhaar_number_valid": true,
    "is_masked": false,
    "name": "Sahil Sahebrao Mankar",
    "dob": "17/11/2001",
    "gender": "MALE",
    "address": null,
    "pin_code": null,
    "document_side": "FRONT"
  },
  "qr_cross_match": {
    "cross_match_status": "QR_NOT_FOUND",
    "match_confidence": 0.0,
    "name_matched": null,
    "dob_matched": null,
    "gender_matched": null,
    "aadhaar_last4_matched": null,
    "pin_code_matched": null,
    "mismatches": [],
    "qr_data": null
  }
}
```

---

## Authenticity Verdict Guide

| Verdict | Meaning | Completeness | Confidence |
| :--- | :--- | :--- | :---: |
| **`AUTHENTIC_COMPLETE_PAIR`** | Front (portrait, emblem, Verhoeff) + Back (Secure QR, address) verified and 100% cross-matched. | `FULL_SET` | **0.95 – 0.99** |
| **`AUTHENTIC_FRONT_ONLY`** | Genuine front card verified; back side/QR was not provided in this upload. | `FRONT_ONLY` | **0.88 – 0.90** |
| **`AUTHENTIC_BACK_ONLY`** | Genuine back card with Secure QR verified; front side/portrait was not provided. | `BACK_ONLY` | **0.88 – 0.92** |
| **`TAMPERING_DETECTED`** | **Fraud Alert 🚨**: Back QR payload directly contradicts Front OCR text (e.g. modified name or number). | Any | **0.98** |
| **`SUSPECTED_FAKE_DOCUMENT`** | **Fraud Alert 🚨**: Plain typed paper, synthetic canvas, or missing card markers/palette/geometry. | `INCOMPLETE` | **0.85** |

---

## Running Automated Tests

```powershell
.venv\Scripts\python -m pytest -v
```

All 21 automated unit and API integration tests execute with 100% pass rate.
