import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import fitz
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ocr_det_model" in data
    assert "ocr_rec_model" in data


def test_upload_image_endpoint(client):
    # Create synthetic test image with balanced brightness (180 is within normal 50-220 range)
    img = np.full((700, 1000, 3), 180, dtype=np.uint8)
    cv2.putText(img, "Government of India", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Ramesh Sharma", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "DOB: 12/04/1995", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Male", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "9999 4105 7036", (80, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    _, enc = cv2.imencode(".png", img)
    img_bytes = enc.tobytes()

    response = client.post(
        "/api/v1/documents/images",
        files={"front_image": ("front.png", img_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "processing_status" in data
    assert len(data["pages"]) == 1
    page = data["pages"][0]
    assert page["image_quality"]["width"] == 1000
    assert page["image_quality"]["height"] == 700
    assert page["image_quality"]["status"] == "GOOD"
    assert "ocr" in page
    assert "lines" in page["ocr"]


def test_upload_pdf_endpoint(client):
    # Create simple 1-page PDF
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((50, 100), "Government of India")
    page.insert_text((50, 150), "9999 4105 7036")
    pdf_bytes = doc.tobytes()
    doc.close()

    response = client.post(
        "/api/v1/documents/pdf",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "processing_status" in data
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1


def test_upload_invalid_image(client):
    response = client.post(
        "/api/v1/documents/images",
        files={"front_image": ("corrupted.jpg", b"not-a-valid-image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unable to decode image" in response.json()["detail"]


def test_upload_empty_file(client):
    response = client.post(
        "/api/v1/documents/images",
        files={"front_image": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_root_redirect_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [302, 307]
    assert response.headers["location"] == "/docs"


def test_openapi_schema(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Aadhaar Validation Service"
    assert "/api/v1/documents/images" in schema["paths"]
    assert "/api/v1/documents/pdf" in schema["paths"]
    assert "/api/v1/documents/test-synthetic" in schema["paths"]
    assert "/api/v1/documents/sample-test-image" in schema["paths"]


def test_sample_image_endpoint(client):
    response = client.get("/api/v1/documents/sample-test-image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 1000


def test_test_synthetic_endpoint(client):
    response = client.post("/api/v1/documents/test-synthetic")
    assert response.status_code == 200
    data = response.json()
    assert data["processing_status"] in ["SUCCESS", "PARTIAL"]
    assert len(data["pages"]) == 1
    assert data["pages"][0]["image_quality"]["status"] == "GOOD"
    assert "ocr" in data["pages"][0]


def test_dependency_injection_swapping(client):
    """
    Demonstrates zero-touch dependency swapping using the dependency-injector container.
    Overrides the heavy PaddleOCR engine with a lightweight mock engine.
    """
    from unittest.mock import MagicMock
    from app.main import app

    mock_ocr = MagicMock()
    mock_ocr.extract.return_value = {
        "status": "SUCCESS",
        "full_text": "Government of India Sneha Patel DOB: 14/08/1992 Female 9999 4105 7036",
        "lines": [
            {"text": "Government of India", "confidence": 0.99, "bbox": [10, 10, 200, 30]},
            {"text": "Sneha Patel", "confidence": 0.98, "bbox": [10, 40, 150, 60]},
            {"text": "DOB: 14/08/1992", "confidence": 0.97, "bbox": [10, 70, 140, 90]},
            {"text": "Female", "confidence": 0.95, "bbox": [10, 100, 80, 120]},
            {"text": "9999 4105 7036", "confidence": 0.99, "bbox": [10, 130, 200, 150]},
        ],
    }

    # Override the OCR singleton in the container
    with app.container.ocr_service.override(mock_ocr):
        response = client.post("/api/v1/documents/test-synthetic")
        assert response.status_code == 200
        data = response.json()
        assert data["aggregated_data"]["aadhaar_number"] == "9999 4105 7036"
        assert data["aggregated_data"]["is_aadhaar_number_valid"] is True
        assert data["aggregated_data"]["dob"] == "14/08/1992"
        assert data["aggregated_data"]["gender"] == "FEMALE"
        mock_ocr.extract.assert_called_once()


def test_request_timing_logged_and_headers(client, caplog):
    """
    Verifies that request execution time is logged and that X-Process-Time is set in headers
    and processing_time_seconds is in the response body.
    """
    import logging
    from unittest.mock import MagicMock
    from app.main import app

    mock_ocr = MagicMock()
    mock_ocr.extract.return_value = {
        "status": "SUCCESS",
        "full_text": "Sample Text",
        "lines": [{"text": "Sample Text", "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
    }

    with caplog.at_level(logging.INFO):
        with app.container.ocr_service.override(mock_ocr):
            response = client.post("/api/v1/documents/test-synthetic")
            assert response.status_code == 200

            # 1. Check HTTP header injected by middleware
            assert "X-Process-Time" in response.headers
            assert response.headers["X-Process-Time"].endswith("s")

            # 2. Check JSON payload timing field
            data = response.json()
            assert "processing_time_seconds" in data
            assert isinstance(data["processing_time_seconds"], (int, float))
            assert data["processing_time_seconds"] >= 0.0

            # 3. Check logs emitted
            log_messages = [rec.message for rec in caplog.records]
            timing_logs = [m for m in log_messages if "completed in" in m or "Completed processing" in m]
            assert len(timing_logs) > 0


def test_holistic_authenticity_response_flags(client):
    """
    Verifies that the main response contains the structured authenticity object with all flags.
    """
    response = client.post("/api/v1/documents/test-synthetic")
    assert response.status_code == 200
    data = response.json()

    assert "authenticity" in data
    auth = data["authenticity"]
    assert "verdict" in auth
    assert "confidence_score" in auth
    assert "completeness" in auth
    assert "is_tampered" in auth
    assert "has_front_face_or_portrait" in auth
    assert "has_back_secure_qr" in auth
    assert "has_valid_verhoeff_checksum" in auth
    assert "has_official_color_palette" in auth
    assert "has_valid_card_geometry" in auth
    assert "security_flags" in auth
    assert isinstance(auth["security_flags"], list)
    assert auth["is_tampered"] is False


def test_front_only_image_authenticity_evaluation(client):
    """
    Verifies that uploading ONLY a front image correctly evaluates to FRONT_ONLY,
    has_back_secure_qr is False, and never claims FULL_SET or SECURE_QR_PRESENT.
    """
    from unittest.mock import MagicMock
    from app.main import app

    mock_ocr = MagicMock()
    mock_ocr.extract.return_value = {
        "status": "SUCCESS",
        "full_text": "Government of India Sneha Patel DOB: 14/08/1992 Female 9999 4105 7036",
        "lines": [
            {"text": "Government of India", "confidence": 0.99, "bbox": [10, 10, 200, 30]},
            {"text": "Sneha Patel", "confidence": 0.98, "bbox": [10, 40, 150, 60]},
            {"text": "DOB: 14/08/1992", "confidence": 0.97, "bbox": [10, 70, 140, 90]},
            {"text": "Female", "confidence": 0.95, "bbox": [10, 100, 80, 120]},
            {"text": "9999 4105 7036", "confidence": 0.99, "bbox": [10, 130, 200, 150]},
        ],
    }

    # Generate synthetic front card image with valid CR80 ratio (1.58) and color palette
    img = np.full((630, 1000, 3), 255, dtype=np.uint8)
    # Saffron stripe
    img[0:80, :] = (40, 130, 240)
    # Green stripe
    img[550:630, :] = (50, 160, 50)
    # Portrait placeholder
    img[120:450, 50:350] = (150, 100, 50)

    _, enc = cv2.imencode(".png", img)
    img_bytes = enc.tobytes()

    with app.container.ocr_service.override(mock_ocr):
        response = client.post(
            "/api/v1/documents/images",
            files={"front_image": ("front.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["document_type"] == "AADHAAR_FRONT"
        assert data["authenticity_status"] == "VERIFIED_CARD_IMAGE"

        auth = data["authenticity"]
        assert auth is not None
        assert auth["completeness"] == "FRONT_ONLY"
        assert auth["verdict"] == "AUTHENTIC_FRONT_ONLY"
        assert auth["has_back_secure_qr"] is False
        assert auth["has_front_face_or_portrait"] is True
        assert auth["has_valid_verhoeff_checksum"] is True
        assert auth["has_official_color_palette"] is True
        assert auth["has_valid_card_geometry"] is True
        assert auth["is_tampered"] is False

        # Security flags must NOT claim back-side security features
        assert "SECURE_QR_PRESENT" not in auth["security_flags"]
        assert "DOCUMENT_FULL_SET_COMPLETE" not in auth["security_flags"]
        assert "PORTRAIT_BIOMETRIC_PRESENT" in auth["security_flags"]
        assert "VERHOEFF_CHECKSUM_VALID" in auth["security_flags"]
        assert "OFFICIAL_PALETTE_DETECTED" in auth["security_flags"]


def test_front_and_back_image_pair_authenticity_evaluation(client):
    """
    Verifies that uploading BOTH front and back images evaluates to FULL_SET and AUTHENTIC_COMPLETE_PAIR.
    """
    from unittest.mock import MagicMock
    from app.main import app

    mock_ocr = MagicMock()
    mock_ocr.extract.side_effect = [
        # Call 1: Front image
        {
            "status": "SUCCESS",
            "full_text": "Government of India Sneha Patel DOB: 14/08/1992 Female 9999 4105 7036",
            "lines": [
                {"text": "Government of India", "confidence": 0.99, "bbox": [10, 10, 200, 30]},
                {"text": "Sneha Patel", "confidence": 0.98, "bbox": [10, 40, 150, 60]},
                {"text": "DOB: 14/08/1992", "confidence": 0.97, "bbox": [10, 70, 140, 90]},
                {"text": "Female", "confidence": 0.95, "bbox": [10, 100, 80, 120]},
                {"text": "9999 4105 7036", "confidence": 0.99, "bbox": [10, 130, 200, 150]},
            ],
        },
        # Call 2: Back image
        {
            "status": "SUCCESS",
            "full_text": "Address: Flat 402, Lotus Tower, MG Road, Bangalore 560001",
            "lines": [
                {"text": "Address: Flat 402, Lotus Tower", "confidence": 0.98, "bbox": [10, 10, 300, 30]},
                {"text": "MG Road, Bangalore 560001", "confidence": 0.97, "bbox": [10, 40, 250, 60]},
            ],
        },
    ]

    real_qr_service = app.container.qr_service()
    mock_qr = MagicMock(wraps=real_qr_service)
    mock_qr.decode_qr_from_image.side_effect = [
        None,  # Front image: no QR
        {      # Back image: decoded QR
            "format": "V2_SECURE_QR",
            "name": "Sneha Patel",
            "dob": "14/08/1992",
            "gender": "F",
            "aadhaar_last4": "7036",
            "pin_code": "560001",
        },
    ]

    # Front image
    front_img = np.full((630, 1000, 3), 255, dtype=np.uint8)
    front_img[0:80, :] = (40, 130, 240)
    front_img[550:630, :] = (50, 160, 50)
    front_img[120:450, 50:350] = (150, 100, 50)
    _, f_enc = cv2.imencode(".png", front_img)

    # Back image
    back_img = np.full((630, 1000, 3), 255, dtype=np.uint8)
    back_img[0:80, :] = (40, 130, 240)
    back_img[550:630, :] = (50, 160, 50)
    _, b_enc = cv2.imencode(".png", back_img)

    with app.container.ocr_service.override(mock_ocr), app.container.qr_service.override(mock_qr):
        response = client.post(
            "/api/v1/documents/images",
            files={
                "front_image": ("front.png", f_enc.tobytes(), "image/png"),
                "back_image": ("back.png", b_enc.tobytes(), "image/png"),
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["authenticity_status"] == "VERIFIED_AUTHENTIC_CARD"
        auth = data["authenticity"]
        assert auth["completeness"] == "FULL_SET"
        assert auth["verdict"] == "AUTHENTIC_COMPLETE_PAIR"
        assert auth["has_front_face_or_portrait"] is True
        assert auth["has_back_secure_qr"] is True
        assert auth["qr_ocr_data_matched"] is True
        assert "DOCUMENT_FULL_SET_COMPLETE" in auth["security_flags"]
        assert "SECURE_QR_PRESENT" in auth["security_flags"]
        assert "QR_DEMOGRAPHICS_MATCHED" in auth["security_flags"]



