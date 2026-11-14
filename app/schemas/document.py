from typing import Literal
from pydantic import BaseModel, Field


class ImageQualityResult(BaseModel):
    width: int = Field(..., description="Image width in pixels", examples=[1200])
    height: int = Field(..., description="Image height in pixels", examples=[780])
    blur_score: float = Field(..., description="Laplacian variance blur score (>50 is generally clear)", examples=[142.35])
    brightness: float = Field(..., description="Average grayscale brightness (0 to 255)", examples=[158.42])
    status: Literal["GOOD", "WARNING"] = Field(..., description="Quality status assessment", examples=["GOOD"])
    warnings: list[str] = Field(
        default_factory=list,
        description="Detected quality warnings (LOW_RESOLUTION, POSSIBLY_BLURRY, TOO_DARK, TOO_BRIGHT)",
    )


class OCRLine(BaseModel):
    text: str = Field(..., description="Recognized line text", examples=["Government of India"])
    confidence: float = Field(ge=0.0, le=1.0, description="OCR recognition confidence (0.0 to 1.0)", examples=[0.9864])
    bbox: list[int] | None = Field(
        None,
        description="Bounding box [xmin, ymin, xmax, ymax] in pixel coordinates",
        examples=[[49, 74, 330, 102]],
    )


class OCRResult(BaseModel):
    status: Literal["SUCCESS", "NO_TEXT"] = Field(..., description="OCR status", examples=["SUCCESS"])
    full_text: str = Field(
        ...,
        description="Concatenated text of all recognized lines",
        examples=["Government of India Rohan Sharma DOB: 12/04/1995 Male 9999 4105 7036"],
    )
    lines: list[OCRLine] = Field(default_factory=list, description="Extracted individual lines with coordinates")


class AadhaarExtractedData(BaseModel):
    aadhaar_number: str | None = Field(None, description="Detected 12-digit Aadhaar number", examples=["9999 4105 7036"])
    is_aadhaar_number_valid: bool | None = Field(None, description="Verhoeff algorithm checksum validation result", examples=[True])
    is_masked: bool = Field(False, description="Whether the number is masked", examples=[False])
    name: str | None = Field(None, description="Extracted individual name", examples=["Rohan Sharma"])
    dob: str | None = Field(None, description="Extracted Date of Birth (DD/MM/YYYY)", examples=["12/04/1995"])
    gender: str | None = Field(None, description="Extracted gender: MALE, FEMALE, or TRANSGENDER", examples=["MALE"])
    address: str | None = Field(None, description="Extracted residential address text")
    pin_code: str | None = Field(None, description="Extracted 6-digit Indian Postal PIN code")
    document_side: Literal["FRONT", "BACK", "FULL_PAGE", "UNKNOWN"] = Field("UNKNOWN", description="Assessed side of card")


class DocumentPageResult(BaseModel):
    page_number: int = Field(..., description="Document page index (1-based)", examples=[1])
    image_quality: ImageQualityResult = Field(..., description="Image quality analysis metrics")
    ocr: OCRResult = Field(..., description="OCR text, lines and bounding box coordinates")
    extracted_data: AadhaarExtractedData | None = Field(None, description="Demographic data for this page")


class DocumentProcessingResponse(BaseModel):
    processing_status: Literal["SUCCESS", "PARTIAL", "FAILED"] = Field(..., description="Overall processing outcome", examples=["SUCCESS"])
    document_type: Literal["AADHAAR_FRONT", "AADHAAR_BACK", "E_AADHAAR", "NOT_CLASSIFIED"] = Field(
        "NOT_CLASSIFIED", description="Classification based on layout and fields"
    )
    authenticity_status: Literal["NOT_VERIFIED"] = Field("NOT_VERIFIED", description="Authenticity flag")
    pages: list[DocumentPageResult] = Field(default_factory=list, description="Per-page image quality and OCR results")
    aggregated_data: AadhaarExtractedData | None = Field(None, description="Combined demographic fields merged across pages")
    message: str = Field(..., description="Status summary message", examples=["Image processing and OCR completed successfully."])
    processing_time_seconds: float = Field(0.0, description="Total document processing time in seconds", examples=[1.25])
