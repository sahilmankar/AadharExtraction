# Aadhaar Validation & OCR Extraction Microservice

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.x-red?style=flat-square)](https://github.com/PaddlePaddle/PaddleOCR)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live-black?style=flat-square&logo=vercel)](https://aadhar-extraction.vercel.app/)

An enterprise-grade, privacy-first microservice for automated Aadhaar parsing, OCR extraction, and multi-factor authenticity validation. Built with **FastAPI**, **PaddleOCR 3.x (PP-OCRv5)**, **OpenCV**, **PyMuPDF**, and **dependency-injector**.

---

## 🌐 Live Production Deployment

The service is deployed and live for immediate testing and integration:

| Resource | URL | Description |
| :--- | :--- | :--- |
| **Interactive Web Portal** | [**aadhar-extraction.vercel.app**](https://aadhar-extraction.vercel.app/) | Production UI to upload card photos/PDFs and inspect extracted demographics & verification diagnostics in real time. |
| **Swagger Interactive Docs** | [**aadhar-extraction.vercel.app/docs**](https://aadhar-extraction.vercel.app/docs) | OpenAPI Swagger interface for testing live endpoints with schemas and interactive parameter execution. |
| **ReDoc Documentation** | [**aadhar-extraction.vercel.app/redoc**](https://aadhar-extraction.vercel.app/redoc) | Clean, comprehensive OpenAPI specification reference. |
| **Service Health Probe** | [**aadhar-extraction.vercel.app/health**](https://aadhar-extraction.vercel.app/health) | Live container status, model cache verification, and worker pool diagnostic check. |

> [!NOTE]
> **Performance Note**: Due to cloud server resource restrictions and cold starts on the hosted demo, requests may take longer to respond. On a local environment or dedicated production hardware, document parsing and validation typically completes in **2–3 seconds**.

---

## 📋 Executive Summary

The **Aadhaar Validation & Extraction Microservice** delivers an automated KYC verification pipeline engineered for banking, fintech, and enterprise onboarding workflows:

- **Zero-PII Disk Retention (Strict Privacy Compliance)**: All card images, PDFs, and extracted resident data are processed strictly in volatile RAM. Unencrypted PII is never persisted to disk, cloud storage, or persistent databases.
- **Three-Way Parallel Processing Architecture**: Every document page concurrently executes **OCR Text Extraction**, **Visual/Format Validation**, and **QR Code Decompression** via asynchronous coroutines (`asyncio.gather`) across a multi-engine worker pool.
- **Multi-Factor Authenticity Triangulation**: Prevents identity fraud and digital tampering by independently verifying:
  1. **Visual Card Geometry**: ISO/IEC 7810 ID-1 (CR80) standard aspect ratio (1.586 ± tolerance), official Government tricolor/maroon palette, and biometric portrait region presence.
  2. **Mathematical Checksum**: UIDAI-standard 12-digit Aadhaar number validation using the **Verhoeff Dihedral Group ($D_{10}$)** algorithm.
  3. **Cryptographic QR Cross-Matching**: Decompresses official UIDAI Secure QR codes (V2/V3 Zlib binary & V1 XML) and cross-checks the cryptographic payload against OCR-extracted demographics (Name, DOB, Gender, Postal PIN, and last 4 digits). Any discrepancy triggers an instant `TAMPERING_DETECTED` security alert.
- **Bilingual Regional Support**: Leverages `devanagari_PP-OCRv5_mobile_rec` to accurately parse dual-language documents (Hindi, Marathi, and other regional scripts) while isolating clean, continuous English address blocks.

---

## 🚀 How to Run & Use

### Method 1: Using the Live Web Portal (Zero Setup)

1. Open [**https://aadhar-extraction.vercel.app/**](https://aadhar-extraction.vercel.app/) in your browser.
2. Select your document type:
   - **Card Photos**: Upload Front and/or Back images (`.jpg`, `.jpeg`, `.png`).
   - **e-Aadhaar PDF**: Upload an unlocked e-Aadhaar PDF document.
   - **Instant Synthetic Test**: Click the amber **"Run Synthetic Test (1-Click)"** button to immediately run a live benchmark without uploading any sensitive files.
3. Review the real-time results:
   - **Identity Card**: Extracted Aadhaar number (with Verhoeff validity check), full name, date of birth, gender, address, and postal PIN code.
   - **Authenticity Assessment**: Holistic verdict (`AUTHENTIC_COMPLETE_PAIR`, `AUTHENTIC_FRONT_ONLY`, etc.), confidence score, and security flags.
   - **QR Cross-Match Table**: Field-by-field audit comparing OCR printed text against the decompressed QR payload.
   - **Raw JSON Response**: Full structured API payload ready for one-click copy.

---

### Method 2: REST API (cURL, Python, JavaScript)

#### 1. Upload Front & Back Card Images
Endpoint: `POST /api/v1/documents/images`

**cURL (Live Vercel)**:
```bash
curl -X POST "https://aadhar-extraction.vercel.app/api/v1/documents/images" \
  -F "front_image=@aadhaar_front.jpg" \
  -F "back_image=@aadhaar_back.jpg"
```

**cURL (Local)**:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/images" \
  -F "front_image=@aadhaar_front.jpg" \
  -F "back_image=@aadhaar_back.jpg"
```

**Python (`requests`)**:
```python
import requests

url = "https://aadhar-extraction.vercel.app/api/v1/documents/images"
files = {
    "front_image": open("aadhaar_front.jpg", "rb"),
    "back_image": open("aadhaar_back.jpg", "rb")  # Optional
}

response = requests.post(url, files=files)
result = response.json()

print(f"Status: {result['processing_status']}")
print(f"Verdict: {result['authenticity']['verdict']}")
print(f"Aadhaar Number: {result['card_data']['aadhaar_number']}")
print(f"Valid Checksum: {result['card_data']['is_aadhaar_number_valid']}")
print(f"Name: {result['card_data']['name']}")
```

---

#### 2. Upload Multi-Page e-Aadhaar PDF
Endpoint: `POST /api/v1/documents/pdf`

**cURL**:
```bash
curl -X POST "https://aadhar-extraction.vercel.app/api/v1/documents/pdf" \
  -F "file=@eaadhaar.pdf"
```

**Python**:
```python
import requests

url = "https://aadhar-extraction.vercel.app/api/v1/documents/pdf"
with open("eaadhaar.pdf", "rb") as f:
    response = requests.post(url, files={"file": f})

data = response.json()
print("Extracted Address:", data["card_data"]["address"])
print("QR Match Status:", data["qr_cross_match"]["cross_match_status"])
```

---

#### 3. 1-Click Instant Synthetic Benchmark
Endpoint: `POST /api/v1/documents/test-synthetic`

Executes OCR parsing, regex extraction, layout detection, and Verhoeff validation instantly without requiring file uploads.

```bash
curl -X POST "https://aadhar-extraction.vercel.app/api/v1/documents/test-synthetic"
```

---

### Method 3: Running with Docker (Recommended for Self-Hosting)

The repository includes a production-ready [`Dockerfile`](Dockerfile) pre-configured with Python 3.11, OpenMP acceleration (`libgomp1`), barcode C-libraries (`libzbar0`), and headless OpenCV (`libgl1`).

```bash
# 1. Build the Docker container image:
docker build -t aadhaar-extractor .

# 2. Run the container:
docker run -d -p 8000:8000 --name aadhaar-service aadhaar-extractor

# 3. Access the service:
# Web Portal: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

---

### Method 4: Local Python Environment Setup

#### Prerequisites
- Python **3.11** (Python 3.11 is required for pre-built PaddlePaddle 3.2 CPU wheels)
- Git

#### Installation Steps

```powershell
# 1. Clone the repository:
git clone https://github.com/sahilmankar/AadharExtraction.git
cd AadharExtraction

# 2. Create and activate a Python 3.11 virtual environment:
python -m venv .venv
.venv\Scripts\activate       # On Windows PowerShell
# source .venv/bin/activate  # On Linux / macOS

# 3. Install dependencies:
pip install -r requirements.txt

# 4. Start the service with hot reload:
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🏗️ Architecture & Component Flow

```
                                  [ Incoming Request ]
                               (Image / e-Aadhaar PDF)
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ Media Ingestion &   │
                               │ In-Memory Buffer    │
                               │ (Zero-Disk Storage) │
                               └──────────┬──────────┘
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   │                      │                      │
                   ▼                      ▼                      ▼
         ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
         │      PART 1      │   │      PART 2      │   │      PART 3      │
         │  PaddleOCR 3.x   │   │ Visual & Quality │   │ pyzbar & Secure  │
         │  (PP-OCRv5 Pool) │   │  Checks (CR80,   │   │  QR Decompressor │
         │   Text Detect &  │   │  Palette, Blur,  │   │   (V1 XML / V2   │
         │    Recognition   │   │    Portrait)     │   │    Zlib Binary)  │
         └─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
                   │                      │                      │
                   └──────────────────────┼──────────────────────┘
                                          │
                                          ▼
                             ┌────────────────────────┐
                             │ Extractor Service      │
                             │ - Verhoeff $D_{10}$    │
                             │ - Bilingual Demographics│
                             │ - Continuous Address   │
                             └────────────┬───────────┘
                                          │
                                          ▼
                             ┌────────────────────────┐
                             │ Holistic Authenticity  │
                             │ & Cross-Match Engine   │
                             │ (OCR vs QR Data Audit) │
                             └────────────┬───────────┘
                                          │
                                          ▼
                             [ Unified JSON Response ]
```

---

## 📊 Response Structure Breakdown

A sample response from `POST /api/v1/documents/images` or `POST /api/v1/documents/pdf`:

```json
{
  "processing_status": "SUCCESS",
  "document_type": "E_AADHAAR",
  "processing_time_seconds": 1.501,

  "card_data": {
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

  "authenticity": {
    "verdict": "AUTHENTIC_COMPLETE_PAIR",
    "confidence_score": 0.99,
    "completeness": "FULL_SET",
    "has_front_face_or_portrait": true,
    "has_secure_qr": true,
    "qr_position": "BACK",
    "has_official_color_palette": true,
    "has_valid_card_geometry": true,
    "security_flags": [
      "PORTRAIT_BIOMETRIC_PRESENT",
      "SECURE_QR_PRESENT",
      "MODERN_BACK_QR_DETECTED",
      "VERHOEFF_CHECKSUM_VALID",
      "OFFICIAL_PALETTE_DETECTED",
      "CARD_GEOMETRY_CR80_VALID",
      "QR_DEMOGRAPHICS_MATCHED",
      "DOCUMENT_FULL_SET_COMPLETE"
    ]
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
      "address": "S/O Sahebrao Mankar, gavthan taluka- ambegaon, Pargaon Tarf Khed, Pune, Maharashtra, 410512",
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
        "full_text": "Government of India Sahil Sahebrao Mankar DOB: 17/11/2001 Male 5134 5777 0655"
      }
    }
  ]
}
```

---

## 🛡️ Authenticity Verdict Reference

| Verdict | Meaning | Completeness | Confidence |
| :--- | :--- | :---: | :---: |
| **`AUTHENTIC_COMPLETE_PAIR`** | Front (portrait, Government palette, Verhoeff checksum) + Back (Secure QR, address block) validated and 100% cross-matched. | `FULL_SET` | **0.95 – 0.99** |
| **`AUTHENTIC_FRONT_ONLY`** | Authentic front card verified; back side/QR was omitted in this upload. | `FRONT_ONLY` | **0.88 – 0.92** |
| **`AUTHENTIC_BACK_ONLY`** | Authentic back card with Secure QR verified; front side was omitted. | `BACK_ONLY` | **0.88 – 0.92** |
| **`TAMPERING_DETECTED`** | **Fraud Alert 🚨**: Cryptographic QR payload directly contradicts printed OCR text (e.g., altered name, forged date of birth, or swapped number). | Any | **0.98** |
| **`SUSPECTED_FAKE_DOCUMENT`** | **Fraud Alert 🚨**: Plain typed paper, synthetic canvas, or missing official card markers/palette/geometry. | `INCOMPLETE` | **0.85** |

---

## 🧪 Automated Testing & Quality Assurance

The test suite covers unit logic, checksum calculations, regex demographic parsing, QR decompression, dependency injection swapping, and endpoint contracts:

```powershell
# Run the test suite:
pytest -v
```

All 26 automated unit and API integration tests execute with a **100% pass rate**.

---

## 📁 Repository Structure

```
.
├── app/
│   ├── main.py                  # FastAPI application lifecycle, CORS, routing
│   ├── api/
│   │   └── document_routes.py   # REST API endpoints (/images, /pdf, /test-synthetic)
│   ├── core/
│   │   ├── config.py            # Pydantic BaseSettings configuration
│   │   └── container.py         # IoC Dependency Injection Container
│   ├── schemas/
│   │   └── document.py          # Pydantic schemas (Authenticity, Quality, OCR, Demographics)
│   ├── services/
│   │   ├── extractor_service.py # Verhoeff checksum, regex extraction & side classification
│   │   ├── media_service.py     # Buffer ingestion, PyMuPDF, OpenCV quality & visual checks
│   │   ├── ocr_service.py       # Engine pool & multi-core PaddleOCR worker
│   │   ├── processing_service.py# 3-way parallel orchestrator & holistic authenticity engine
│   │   └── qr_service.py        # Dual-format QR decoder (V2 Zlib & V1 XML) + Cross-match
│   └── static/
│       └── index.html           # Enterprise single-page web portal (Tailwind CSS, Lucide Icons)
├── models/
│   └── official_models/         # Pre-provisioned local PaddleOCR deep learning models
├── tests/
│   ├── test_api.py              # API integration, DI swapping, and timing tests
│   └── test_extraction.py       # Verhoeff, regex, visual authenticity & QR tests
├── Dockerfile                   # Production container definition (Python 3.11-slim)
├── .dockerignore                # Build-context optimization exclusions
├── requirements.txt             # Pinned dependency definitions
└── README.md                    # Project documentation & integration guide
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
