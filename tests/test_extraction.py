import pytest
import numpy as np
import cv2
import fitz
from app.services.extractor_service import validate_verhoeff, AadhaarExtractorService
from app.services.file_service import FileService, InvalidDocumentError
from app.services.image_service import ImageQualityService
from app.schemas.document import OCRLine


def test_verhoeff_validation():
    # Valid Aadhaar numbers satisfy Verhoeff checksum
    # Example: 9999 4105 7036
    clean = "999941057036"
    assert validate_verhoeff(clean) is True

    # Invalid checksum
    invalid = "999941057032"
    assert validate_verhoeff(invalid) is False

    # Short number
    assert validate_verhoeff("12345") is False


def test_extractor_service_aadhaar_parsing():
    extractor = AadhaarExtractorService()
    lines = [
        OCRLine(text="भारत सरकार", confidence=0.99, bbox=[50, 20, 200, 50]),
        OCRLine(text="Government of India", confidence=0.98, bbox=[50, 55, 300, 85]),
        OCRLine(text="Suresh Kumar", confidence=0.95, bbox=[50, 120, 220, 150]),
        OCRLine(text="जन्म तारीख / DOB: 15/08/1990", confidence=0.97, bbox=[50, 160, 320, 190]),
        OCRLine(text="पुरुष / MALE", confidence=0.99, bbox=[50, 200, 180, 230]),
        OCRLine(text="9999 4105 7036", confidence=0.99, bbox=[50, 260, 280, 290]),
    ]
    full_text = " ".join([l.text for l in lines])

    extracted = extractor.extract_from_lines(lines, full_text)
    assert extracted.aadhaar_number == "9999 4105 7036"
    assert extracted.is_aadhaar_number_valid is True
    assert extracted.dob == "15/08/1990"
    assert extracted.gender == "MALE"
    assert extracted.name == "Suresh Kumar"
    assert extracted.document_side == "FRONT"

    classification = extractor.classify_document(extracted, full_text)
    assert classification == "AADHAAR_FRONT"


def test_extractor_service_back_side_parsing():
    extractor = AadhaarExtractorService()
    lines = [
        OCRLine(text="Unique Identification Authority of India", confidence=0.98, bbox=[50, 20, 400, 50]),
        OCRLine(text="Address: S/O Ramesh Kumar, Flat 201, Green Valley", confidence=0.95, bbox=[50, 80, 450, 110]),
        OCRLine(text="Sector 62, Noida, Uttar Pradesh, 201301", confidence=0.96, bbox=[50, 115, 420, 145]),
        OCRLine(text="9999 4105 7033", confidence=0.99, bbox=[50, 260, 280, 290]),
    ]
    full_text = " ".join([l.text for l in lines])

    extracted = extractor.extract_from_lines(lines, full_text)
    assert extracted.pin_code == "201301"
    assert "Flat 201" in (extracted.address or "")
    assert extracted.document_side == "BACK"

    classification = extractor.classify_document(extracted, full_text)
    assert classification == "AADHAAR_BACK"


def test_image_quality_service():
    quality_service = ImageQualityService()

    # Create synthetic sharp image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(img, "Sharp Sample Text", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    result = quality_service.analyze(img)
    assert result["width"] == 800
    assert result["height"] == 600
    assert "warnings" in result
    assert result["blur_score"] > 0

    # Low resolution image
    small_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res_small = quality_service.analyze(small_img)
    assert "LOW_RESOLUTION" in res_small["warnings"]
    assert res_small["status"] == "WARNING"


def test_file_service_pdf_and_image():
    file_service = FileService(
        max_upload_bytes=5 * 1024 * 1024,
        max_image_pixels=20_000_000,
        max_pdf_pages=5,
        pdf_render_dpi=150,
    )

    # 1. Test image decoding
    test_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, enc = cv2.imencode(".png", test_img)
    decoded = file_service.decode_image(enc.tobytes())
    assert decoded.shape == (100, 100, 3)

    # 2. Test PDF rendering
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((50, 50), "Aadhaar Test PDF")
    pdf_bytes = doc.tobytes()
    doc.close()

    images = file_service.pdf_to_images(pdf_bytes)
    assert len(images) == 1
    assert images[0].shape[0] > 0
    assert images[0].shape[1] > 0
