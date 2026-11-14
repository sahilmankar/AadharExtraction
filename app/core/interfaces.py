from typing import Protocol, runtime_checkable
import numpy as np
from fastapi import UploadFile
from app.schemas.document import AadhaarExtractedData, DocumentProcessingResponse, OCRLine


@runtime_checkable
class IMediaService(Protocol):
    """Protocol for reading uploads, rasterizing PDFs, quality diagnostics, and ROI isolation."""
    async def read_upload(self, upload: UploadFile) -> bytes: ...
    def decode_image(self, data: bytes) -> np.ndarray: ...
    def pdf_to_images(self, data: bytes) -> list[np.ndarray]: ...
    def prepare_page_image(self, raw_image: np.ndarray) -> tuple[dict, np.ndarray]: ...


# Compatibility aliases
IFileService = IMediaService
IImageQualityService = IMediaService


@runtime_checkable
class IOCREngine(Protocol):
    """Protocol for OCR inference engines."""
    def extract(self, image: np.ndarray) -> dict: ...


@runtime_checkable
class IAadhaarExtractor(Protocol):
    """Protocol for regex, layout extraction, and Verhoeff validation."""
    def extract_from_lines(self, lines: list[OCRLine], full_text: str) -> AadhaarExtractedData: ...
    def merge_extracted_data(self, data_list: list[AadhaarExtractedData]) -> AadhaarExtractedData: ...
    def classify_document(self, merged: AadhaarExtractedData, full_text: str) -> str: ...


@runtime_checkable
class IDocumentProcessingService(Protocol):
    """Protocol for document processing orchestrator."""
    async def process_images(self, images: list[np.ndarray]) -> DocumentProcessingResponse: ...
    async def process_image_upload(
        self, front_image: UploadFile, back_image: UploadFile | None = None
    ) -> DocumentProcessingResponse: ...
    async def process_pdf_upload(self, file: UploadFile) -> DocumentProcessingResponse: ...
