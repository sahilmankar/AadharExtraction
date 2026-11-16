from typing import Literal
from pydantic import BaseModel, Field


class ImageQualityResult(BaseModel):
    width: int
    height: int
    blur_score: float
    brightness: float
    status: Literal["GOOD", "WARNING"]
    warnings: list[str] = Field(default_factory=list)


class VisualAuthenticityResult(BaseModel):
    is_valid_aadhaar_card_image: bool
    aspect_ratio: float
    qr_detected: bool
    has_aadhaar_palette: bool
    has_photo_or_graphics: bool
    checks_passed: list[str] = Field(default_factory=list)


class OCRLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] | None = None


class OCRResult(BaseModel):
    status: Literal["SUCCESS", "NO_TEXT"]
    full_text: str
    lines: list[OCRLine] = Field(default_factory=list)


class AadhaarExtractedData(BaseModel):
    aadhaar_number: str | None = None
    is_aadhaar_number_valid: bool | None = None
    is_masked: bool = False
    name: str | None = None
    dob: str | None = None
    gender: str | None = None
    address: str | None = None
    pin_code: str | None = None
    document_side: Literal["FRONT", "BACK", "FULL_PAGE", "UNKNOWN"] = "UNKNOWN"


class QRExtractedData(BaseModel):
    format: str
    reference_id: str | None = None
    aadhaar_last4: str | None = None
    name: str | None = None
    dob: str | None = None
    gender: str | None = None
    care_of: str | None = None
    pin_code: str | None = None
    district: str | None = None
    state: str | None = None
    address: str | None = None
    has_photo: bool = False


class QRCrossMatchResult(BaseModel):
    cross_match_status: Literal["MATCHED", "MISMATCH_DETECTED", "QR_NOT_FOUND", "PAYLOAD_UNREADABLE_LOW_RESOLUTION"]
    match_confidence: float = Field(ge=0.0, le=1.0)
    name_matched: bool | None = None
    dob_matched: bool | None = None
    gender_matched: bool | None = None
    aadhaar_last4_matched: bool | None = None
    pin_code_matched: bool | None = None
    mismatches: list[str] = Field(default_factory=list)
    qr_data: QRExtractedData | None = None


class DocumentPageResult(BaseModel):
    page_number: int
    image_quality: ImageQualityResult
    visual_authenticity: VisualAuthenticityResult | None = None
    ocr: OCRResult


class DocumentAuthenticityResult(BaseModel):
    verdict: Literal[
        "AUTHENTIC_COMPLETE_PAIR",
        "AUTHENTIC_FRONT_ONLY",
        "AUTHENTIC_BACK_ONLY",
        "TAMPERING_DETECTED",
        "SUSPECTED_FAKE_DOCUMENT",
        "NOT_VERIFIED",
    ]
    confidence_score: float = Field(ge=0.0, le=1.0)
    completeness: Literal["FULL_SET", "FRONT_ONLY", "BACK_ONLY", "INCOMPLETE"]
    has_front_face_or_portrait: bool = False
    has_secure_qr: bool = False
    qr_position: Literal["FRONT", "BACK", "NOT_FOUND"] = "NOT_FOUND"
    has_official_color_palette: bool = False
    has_valid_card_geometry: bool = False
    security_flags: list[str] = Field(default_factory=list)


class DocumentProcessingResponse(BaseModel):
    processing_status: Literal["SUCCESS", "PARTIAL", "FAILED"]
    document_type: Literal["AADHAAR_FRONT", "AADHAAR_BACK", "E_AADHAAR", "NOT_CLASSIFIED"] = "NOT_CLASSIFIED"
    card_data: AadhaarExtractedData | None = None
    authenticity: DocumentAuthenticityResult | None = None
    qr_cross_match: QRCrossMatchResult | None = None
    pages: list[DocumentPageResult] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
