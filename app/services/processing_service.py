import asyncio
import logging
import time
from typing import Any
import numpy as np
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.document import (
    AadhaarExtractedData,
    DocumentAuthenticityResult,
    DocumentPageResult,
    DocumentProcessingResponse,
    ImageQualityResult,
    OCRLine,
    OCRResult,
    QRCrossMatchResult,
    QRExtractedData,
    VisualAuthenticityResult,
)
from app.services.extractor_service import AadhaarExtractorService
from app.services.media_service import MediaService
from app.services.ocr_service import OCRService
from app.services.qr_service import AadhaarQRService

logger = logging.getLogger(__name__)


def _resolve_completeness(pages_count: int, doc_type: str, is_front: bool, is_back: bool) -> str:
    """Calculates whether the document represents a FULL_SET, single side, or INCOMPLETE."""
    if pages_count == 1 and doc_type != "E_AADHAAR":
        if (is_front and not is_back) or doc_type == "AADHAAR_FRONT":
            return "FRONT_ONLY"
        if (is_back and not is_front) or doc_type == "AADHAAR_BACK":
            return "BACK_ONLY"
        return "INCOMPLETE"
    if doc_type == "E_AADHAAR" or (is_front and is_back):
        return "FULL_SET"
    return "FRONT_ONLY" if is_front else ("BACK_ONLY" if is_back else "INCOMPLETE")


def _resolve_verdict(
    completeness: str,
    is_tampered: bool,
    qr_matched: bool | None,
    palette: bool,
    verhoeff: bool,
    portrait: bool,
    secure_qr: bool,
    back_data: bool,
) -> tuple[str, float]:
    """Pure decision matrix computing the confidence score and authenticity verdict."""
    if is_tampered:
        return "TAMPERING_DETECTED", 0.98
    if completeness == "FULL_SET":
        if qr_matched is True and verhoeff:
            return "AUTHENTIC_COMPLETE_PAIR", 0.99
        if palette and verhoeff:
            return "AUTHENTIC_COMPLETE_PAIR", 0.92
        return "AUTHENTIC_COMPLETE_PAIR", 0.88
    if completeness == "FRONT_ONLY":
        if palette and verhoeff:
            return "AUTHENTIC_FRONT_ONLY", 0.90
        if (portrait or palette) and verhoeff:
            return "AUTHENTIC_FRONT_ONLY", 0.88
        return "AUTHENTIC_FRONT_ONLY", 0.80
    if completeness == "BACK_ONLY":
        if secure_qr and qr_matched is not False:
            return "AUTHENTIC_BACK_ONLY", 0.92
        return "AUTHENTIC_BACK_ONLY", 0.85
    if not palette and not portrait and not secure_qr:
        return "SUSPECTED_FAKE_DOCUMENT", 0.85
    return "NOT_VERIFIED", 0.50


class DocumentProcessingService:
    """Concurrent document processing orchestrator managing OCR, QR decoding, and authenticity."""

    def __init__(
        self,
        media_service: MediaService,
        ocr_service: OCRService,
        extractor_service: AadhaarExtractorService,
        qr_service: AadhaarQRService | None = None,
    ):
        self.media_service = media_service
        self.ocr_service = ocr_service
        self.extractor_service = extractor_service
        self.qr_service = qr_service or AadhaarQRService()

    async def _process_single_page(self, page_num: int, raw_image: np.ndarray) -> tuple[DocumentPageResult, dict | None]:
        t_start = time.perf_counter()
        quality_dict, roi_img = await run_in_threadpool(self.media_service.prepare_page_image, raw_image)

        async def _ocr_task():
            data = await run_in_threadpool(self.ocr_service.extract, roi_img)
            lines = [OCRLine(**line) for line in data["lines"]]
            return (
                OCRResult(status=data["status"], full_text=data["full_text"], lines=lines),
                self.extractor_service.extract_from_lines(lines, data["full_text"]),
            )

        (ocr_result, extracted), visual_dict, qr_data = await asyncio.gather(
            _ocr_task(),
            run_in_threadpool(self.media_service.analyze_visual_authenticity, roi_img),
            run_in_threadpool(self.qr_service.decode_qr_from_image, raw_image),
        )

        logger.info("Page %d processed in %.3fs", page_num, time.perf_counter() - t_start)
        page_result = DocumentPageResult(
            page_number=page_num,
            image_quality=ImageQualityResult(**quality_dict),
            visual_authenticity=VisualAuthenticityResult(**visual_dict),
            ocr=ocr_result,
            extracted_data=extracted,
        )
        return page_result, qr_data

    async def process_images(self, images: list[np.ndarray]) -> DocumentProcessingResponse:
        t_start = time.perf_counter()

        # Process all pages concurrently
        results = await asyncio.gather(*[self._process_single_page(i, img) for i, img in enumerate(images, start=1)])
        pages = sorted([r[0] for r in results], key=lambda p: p.page_number)

        # Merge extracted text attributes and classify document side
        extracted_list = [p.extracted_data for p in pages if p.extracted_data is not None]
        aggregated_data = self.extractor_service.merge_extracted_data(extracted_list)
        combined_text = " ".join(p.ocr.full_text for p in pages)
        doc_type = self.extractor_service.classify_document(aggregated_data, combined_text)

        # First decoded QR across pages & cross-match
        discovered_qr = next((r[1] for r in results if r[1] is not None), None)
        cross_match_dict = self.qr_service.cross_match(aggregated_data, discovered_qr)
        qr_obj = QRExtractedData(**cross_match_dict["qr_data"]) if cross_match_dict.get("qr_data") else None
        qr_cross_match = QRCrossMatchResult(
            cross_match_status=cross_match_dict["cross_match_status"],
            match_confidence=cross_match_dict["match_confidence"],
            name_matched=cross_match_dict["name_matched"],
            dob_matched=cross_match_dict["dob_matched"],
            gender_matched=cross_match_dict["gender_matched"],
            aadhaar_last4_matched=cross_match_dict["aadhaar_last4_matched"],
            pin_code_matched=cross_match_dict["pin_code_matched"],
            mismatches=cross_match_dict["mismatches"],
            qr_data=qr_obj,
        )

        auth_result = self._evaluate_holistic_authenticity(pages, aggregated_data, discovered_qr, qr_cross_match, doc_type)

        status_mapping = {
            "TAMPERING_DETECTED": "TAMPERING_DETECTED",
            "AUTHENTIC_COMPLETE_PAIR": "VERIFIED_AUTHENTIC_CARD",
            "AUTHENTIC_FRONT_ONLY": "VERIFIED_CARD_IMAGE",
            "AUTHENTIC_BACK_ONLY": "VERIFIED_CARD_IMAGE",
            "SUSPECTED_FAKE_DOCUMENT": "SUSPECTED_NON_CARD",
        }
        authenticity = status_mapping.get(auth_result.verdict, "NOT_VERIFIED")
        status = "SUCCESS" if any(p.ocr.status == "SUCCESS" for p in pages) else "PARTIAL"
        duration = time.perf_counter() - t_start

        logger.info("Processed %d page(s) in %.3fs [Verdict: %s | Score: %.2f]", len(pages), duration, auth_result.verdict, auth_result.confidence_score)

        return DocumentProcessingResponse(
            processing_status=status,
            document_type=doc_type,
            authenticity_status=authenticity,
            authenticity=auth_result,
            pages=pages,
            aggregated_data=aggregated_data,
            qr_cross_match=qr_cross_match,
            message="Image processing, OCR, and multi-factor authenticity verification completed successfully.",
            processing_time_seconds=round(duration, 3),
        )

    def _evaluate_holistic_authenticity(
        self,
        pages: list[DocumentPageResult],
        aggregated_data: AadhaarExtractedData,
        discovered_qr: dict | None,
        qr_cross_match: QRCrossMatchResult,
        doc_type: str,
    ) -> DocumentAuthenticityResult:
        security_flags = []

        has_portrait = any(p.visual_authenticity and p.visual_authenticity.has_photo_or_graphics for p in pages)
        is_front_present = has_portrait or bool(aggregated_data.name or aggregated_data.dob) or (doc_type == "AADHAAR_FRONT")
        if has_portrait:
            security_flags.append("PORTRAIT_BIOMETRIC_PRESENT")

        has_secure_qr = discovered_qr is not None
        has_back_data = bool(aggregated_data.address)
        is_back_present = has_secure_qr or has_back_data or (doc_type == "AADHAAR_BACK")
        if has_secure_qr:
            security_flags.append("SECURE_QR_PRESENT")

        has_valid_verhoeff = bool(aggregated_data.is_aadhaar_number_valid)
        if has_valid_verhoeff:
            security_flags.append("VERHOEFF_CHECKSUM_VALID")

        has_palette = any(p.visual_authenticity and p.visual_authenticity.has_aadhaar_palette for p in pages)
        if has_palette:
            security_flags.append("OFFICIAL_PALETTE_DETECTED")

        has_geometry = any(p.visual_authenticity and (1.20 <= p.visual_authenticity.aspect_ratio <= 1.90) for p in pages)
        if has_geometry:
            security_flags.append("CARD_GEOMETRY_CR80_VALID")

        is_tampered = qr_cross_match.cross_match_status == "MISMATCH_DETECTED"
        qr_matched = None
        if discovered_qr is not None:
            if qr_cross_match.cross_match_status == "MATCHED":
                qr_matched = True
                security_flags.append("QR_DEMOGRAPHICS_MATCHED")
            elif is_tampered:
                qr_matched = False
                security_flags.append("TAMPERING_MISMATCH_DETECTED")

        completeness = _resolve_completeness(len(pages), doc_type, is_front_present, is_back_present)
        if completeness == "FULL_SET":
            security_flags.append("DOCUMENT_FULL_SET_COMPLETE")

        verdict, confidence = _resolve_verdict(
            completeness=completeness,
            is_tampered=is_tampered,
            qr_matched=qr_matched,
            palette=has_palette,
            verhoeff=has_valid_verhoeff,
            portrait=has_portrait,
            secure_qr=has_secure_qr,
            back_data=has_back_data,
        )
        if verdict == "SUSPECTED_FAKE_DOCUMENT":
            security_flags.append("SUSPECTED_NON_CARD_CANVAS")

        return DocumentAuthenticityResult(
            verdict=verdict,
            confidence_score=confidence,
            completeness=completeness,
            is_tampered=is_tampered,
            has_front_face_or_portrait=has_portrait,
            has_back_secure_qr=has_secure_qr,
            has_valid_verhoeff_checksum=has_valid_verhoeff,
            has_official_color_palette=has_palette,
            has_valid_card_geometry=has_geometry,
            qr_ocr_data_matched=qr_matched,
            security_flags=security_flags,
        )

    async def process_image_upload(
        self, front_image: UploadFile, back_image: UploadFile | None = None
    ) -> DocumentProcessingResponse:
        images = [self.media_service.decode_image(await self.media_service.read_upload(front_image))]
        if back_image and back_image.filename:
            images.append(self.media_service.decode_image(await self.media_service.read_upload(back_image)))
        return await self.process_images(images)

    async def process_pdf_upload(self, file: UploadFile) -> DocumentProcessingResponse:
        pdf_bytes = await self.media_service.read_upload(file)
        return await self.process_images(self.media_service.pdf_to_images(pdf_bytes))
