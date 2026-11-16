import pytest
import numpy as np
import cv2
import fitz
from app.services.extractor_service import validate_verhoeff, AadhaarExtractorService
from app.services.media_service import MediaService, InvalidDocumentError
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
    quality_service = MediaService()

    # Create synthetic sharp image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(img, "Sharp Sample Text", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    result = quality_service.analyze_quality(img)
    assert result["width"] == 800
    assert result["height"] == 600
    assert "warnings" in result
    assert result["blur_score"] > 0

    # Low resolution image
    small_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res_small = quality_service.analyze_quality(small_img)
    assert "LOW_RESOLUTION" in res_small["warnings"]
    assert res_small["status"] == "WARNING"


def test_visual_authenticity_verification():
    from app.services.media_service import MediaService

    media = MediaService()

    # 1. Plain text / synthetic white canvas (Fake document)
    fake_plain = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(fake_plain, "Plain Typed Aadhaar: 9999 4105 7036", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    fake_result = media.analyze_visual_authenticity(fake_plain)
    assert fake_result["is_valid_aadhaar_card_image"] is False
    assert fake_result["qr_detected"] is False
    assert fake_result["has_aadhaar_palette"] is False

    # 2. Card with Aadhaar colors and aspect ratio
    card = np.full((600, 950, 3), 245, dtype=np.uint8)  # Aspect ratio ~1.58
    # Add saffron top stripe (HSV [15, 200, 240])
    saffron_bgr = [40, 160, 255]
    card[20:70, 50:900] = saffron_bgr
    # Add green bottom stripe (HSV [60, 200, 150])
    green_bgr = [30, 180, 50]
    card[530:570, 50:900] = green_bgr
    # Add maroon logo square
    maroon_bgr = [30, 20, 180]
    card[80:150, 800:870] = maroon_bgr

    card_result = media.analyze_visual_authenticity(card)
    assert card_result["is_valid_aadhaar_card_image"] is True
    assert card_result["has_aadhaar_palette"] is True
    assert "VALID_ASPECT_RATIO" in card_result["checks_passed"]
    assert "AADHAAR_PALETTE_DETECTED" in card_result["checks_passed"]


def test_file_service_pdf_and_image():
    file_service = MediaService(
        max_upload_bytes=5 * 1024 * 1024,
        max_image_pixels=20_000_000,
        max_pdf_pages=5,
        pdf_render_dpi=120,
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


def test_qr_service_parsing_and_cross_match():
    from app.services.qr_service import AadhaarQRService
    from app.schemas.document import AadhaarExtractedData

    qr_service = AadhaarQRService()

    # 1. Test Legacy XML QR parsing
    xml_payload = b'<PrintLetterBarcodeData uid="999941057036" name="Ramesh Sharma" gender="M" dob="12/04/1995" dist="Pune" pc="411001" state="Maharashtra" />'
    parsed_xml = qr_service.parse_qr_payload(xml_payload)
    assert parsed_xml is not None
    assert parsed_xml["format"] == "XML_V1"
    assert parsed_xml["name"] == "Ramesh Sharma"
    assert parsed_xml["aadhaar_last4"] == "7036"
    assert parsed_xml["dob"] == "12/04/1995"
    assert parsed_xml["pin_code"] == "411001"

    # 2. Test Cross-Match (MATCHED)
    ocr_data = AadhaarExtractedData(
        aadhaar_number="9999 4105 7036",
        name="Ramesh Sharma",
        dob="12/04/1995",
        gender="MALE",
        pin_code="411001",
    )
    match_res = qr_service.cross_match(ocr_data, parsed_xml)
    assert match_res["cross_match_status"] == "MATCHED"
    assert match_res["match_confidence"] == 1.0
    assert match_res["name_matched"] is True
    assert match_res["dob_matched"] is True
    assert match_res["aadhaar_last4_matched"] is True
    assert len(match_res["mismatches"]) == 0

    # 3. Test Cross-Match Tampering Detection (MISMATCH)
    tampered_ocr_data = AadhaarExtractedData(
        aadhaar_number="9999 4105 9999",  # Fake number
        name="Vikram Singh",              # Fake name
        dob="12/04/1995",
        gender="MALE",
        pin_code="411001",
    )
    mismatch_res = qr_service.cross_match(tampered_ocr_data, parsed_xml)
    assert mismatch_res["cross_match_status"] == "MISMATCH_DETECTED"
    assert mismatch_res["name_matched"] is False
    assert mismatch_res["aadhaar_last4_matched"] is False
    assert len(mismatch_res["mismatches"]) >= 2
