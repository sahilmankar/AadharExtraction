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


# ---------------------------------------------------------------------------
# Decision Matrix Confidence Scores
# ---------------------------------------------------------------------------
# 0.99: Perfect cryptographic match (QR verified against OCR text + valid Verhoeff checksum)
CONF_PAIR_QR_MATCHED = 0.99
# 0.98: High-confidence fraud alert (OCR demographic text contradicts digitally signed QR data)
CONF_TAMPERING_DETECTED = 0.98
# 0.95: Legacy front-only card with readable QR and valid Verhoeff checksum
CONF_FRONT_QR_MATCHED = 0.95
# 0.92: Complete pair (front + back) possessing official tri-color palette and Verhoeff checksum
CONF_PAIR_PALETTE_VERHOEFF = 0.92
# 0.90: Single front card possessing official tri-color palette and Verhoeff checksum
CONF_FRONT_PALETTE_VERHOEFF = 0.90
# 0.88: Complete pair or front side possessing portrait photo and Verhoeff checksum
CONF_FRONT_PORTRAIT_VERHOEFF = 0.88
# 0.80: Basic single front side without verified palette or portrait
CONF_FRONT_BASIC = 0.80
# 0.92: Back card with valid decoded or detected secure QR code
CONF_BACK_SECURE_QR = 0.92
# 0.85: Back card with address text only
CONF_BACK_BASIC = 0.85
# 0.85: Suspected fake document (lacks official palette, portrait photo, and QR code)
CONF_SUSPECTED_FAKE = 0.85
# 0.50: Ambiguous or unverified document
CONF_UNVERIFIED = 0.50

# Aspect ratio bounds for ISO/IEC 7810 ID-1 standard card (nominal: 1.586)
# Allowing +/- 20% margin for perspective angle and lens distortion:
ASPECT_RATIO_CR80_MIN = 1.20
ASPECT_RATIO_CR80_MAX = 1.90


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
        return "TAMPERING_DETECTED", CONF_TAMPERING_DETECTED
    if completeness == "FULL_SET":
        if qr_matched is True and verhoeff:
            return "AUTHENTIC_COMPLETE_PAIR", CONF_PAIR_QR_MATCHED
        if palette and verhoeff:
            return "AUTHENTIC_COMPLETE_PAIR", CONF_PAIR_PALETTE_VERHOEFF
        return "AUTHENTIC_COMPLETE_PAIR", CONF_FRONT_PORTRAIT_VERHOEFF
    if completeness == "FRONT_ONLY":
        if qr_matched is True and verhoeff:
            return "AUTHENTIC_FRONT_ONLY", CONF_FRONT_QR_MATCHED
        if palette and verhoeff:
            return "AUTHENTIC_FRONT_ONLY", CONF_FRONT_PALETTE_VERHOEFF
        if (portrait or palette) and verhoeff:
            return "AUTHENTIC_FRONT_ONLY", CONF_FRONT_PORTRAIT_VERHOEFF
        return "AUTHENTIC_FRONT_ONLY", CONF_FRONT_BASIC
    if completeness == "BACK_ONLY":
        if secure_qr and qr_matched is not False:
            return "AUTHENTIC_BACK_ONLY", CONF_BACK_SECURE_QR
        return "AUTHENTIC_BACK_ONLY", CONF_BACK_BASIC
    if not palette and not portrait and not secure_qr:
        return "SUSPECTED_FAKE_DOCUMENT", CONF_SUSPECTED_FAKE
    return "NOT_VERIFIED", CONF_UNVERIFIED


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

    async def _process_single_page(
        self, page_num: int, raw_image: np.ndarray
    ) -> tuple[DocumentPageResult, dict | None, AadhaarExtractedData]:
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
        )
        return page_result, qr_data, extracted

    async def process_images(self, images: list[np.ndarray]) -> DocumentProcessingResponse:
        t_start = time.perf_counter()

        # Process all pages concurrently
        results = await asyncio.gather(*[self._process_single_page(i, img) for i, img in enumerate(images, start=1)])
        pages = sorted([r[0] for r in results], key=lambda p: p.page_number)

        # Merge extracted text attributes and classify document side
        extracted_list = [r[2] for r in results if r[2] is not None]
        aggregated_data = self.extractor_service.merge_extracted_data(extracted_list)
        combined_text = " ".join(p.ocr.full_text for p in pages)
        doc_type = self.extractor_service.classify_document(aggregated_data, combined_text)

        # First decoded QR across pages & cross-match
        discovered_qr = next((r[1] for r in results if r[1] is not None), None)
        has_visual_qr = any(p.visual_authenticity and p.visual_authenticity.qr_detected for p in pages)
        cross_match_dict = self.qr_service.cross_match(aggregated_data, discovered_qr, has_visual_qr=has_visual_qr)
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

        auth_result = self._evaluate_holistic_authenticity(pages, results, aggregated_data, discovered_qr, qr_cross_match, doc_type)

        status = "SUCCESS" if any(p.ocr.status == "SUCCESS" for p in pages) else "PARTIAL"
        duration = time.perf_counter() - t_start

        logger.info("Processed %d page(s) in %.3fs [Verdict: %s | Score: %.2f]", len(pages), duration, auth_result.verdict, auth_result.confidence_score)

        return DocumentProcessingResponse(
            processing_status=status,
            document_type=doc_type,
            card_data=aggregated_data,
            authenticity=auth_result,
            qr_cross_match=qr_cross_match,
            pages=pages,
            processing_time_seconds=round(duration, 3),
        )

    def _evaluate_holistic_authenticity(
        self,
        pages: list[DocumentPageResult],
        results: list[tuple[DocumentPageResult, dict | None, AadhaarExtractedData]],
        aggregated_data: AadhaarExtractedData,
        discovered_qr: dict | None,
        qr_cross_match: QRCrossMatchResult,
        doc_type: str,
    ) -> DocumentAuthenticityResult:
        security_flags = []

        # 1. Front markers
        has_portrait = any(p.visual_authenticity and p.visual_authenticity.has_photo_or_graphics for p in pages)
        is_front_present = has_portrait or bool(aggregated_data.name or aggregated_data.dob) or (doc_type == "AADHAAR_FRONT")
        if has_portrait:
            security_flags.append("PORTRAIT_BIOMETRIC_PRESENT")

        # 2. QR Code Position & Format Detection (Supports Old Cards with Front QR and Modern Cards with Back QR)
        has_decoded_qr = discovered_qr is not None
        has_visual_qr = any(p.visual_authenticity and p.visual_authenticity.qr_detected for p in pages)
        has_secure_qr = has_decoded_qr or has_visual_qr
        qr_position = "NOT_FOUND"
        qr_on_front = False
        qr_on_back = False

        target_page = None
        target_extracted = None
        if has_decoded_qr:
            security_flags.append("SECURE_QR_PRESENT")
            qr_page_entry = next((r for r in results if r[1] is not None), None)
            if qr_page_entry is not None:
                target_page = qr_page_entry[0]
                target_extracted = qr_page_entry[2]
        elif has_visual_qr:
            security_flags.append("SECURE_QR_PATTERN_PRESENT")
            vis_entry = next((r for r in results if r[0].visual_authenticity and r[0].visual_authenticity.qr_detected), None)
            if vis_entry is not None:
                target_page = vis_entry[0]
                target_extracted = vis_entry[2]

        if target_page is not None:
            p_side = target_extracted.document_side if target_extracted else "UNKNOWN"
            # Page is front if marked FRONT, or has Name/DOB and no address, or is Page 1 without address
            is_on_front = (
                (p_side == "FRONT")
                or (target_page.page_number == 1 and not (target_extracted and target_extracted.address))
                or (target_extracted and (target_extracted.name or target_extracted.dob) and not target_extracted.address)
                or (target_page.visual_authenticity and target_page.visual_authenticity.has_photo_or_graphics and not (target_extracted and target_extracted.address))
            )

            if is_on_front:
                qr_position = "FRONT"
                qr_on_front = True
                security_flags.append("LEGACY_FRONT_QR_DETECTED")
            else:
                qr_position = "BACK"
                qr_on_back = True
                security_flags.append("MODERN_BACK_QR_DETECTED")

        # 3. Back markers (Address or QR on back side)
        has_back_data = bool(aggregated_data.address)
        is_back_present = has_back_data or qr_on_back or (doc_type == "AADHAAR_BACK")

        # 4. Checksums & Geometry
        has_valid_verhoeff = bool(aggregated_data.is_aadhaar_number_valid)
        if has_valid_verhoeff:
            security_flags.append("VERHOEFF_CHECKSUM_VALID")

        has_palette = any(p.visual_authenticity and p.visual_authenticity.has_aadhaar_palette for p in pages)
        if has_palette:
            security_flags.append("OFFICIAL_PALETTE_DETECTED")

        has_geometry = any(p.visual_authenticity and (ASPECT_RATIO_CR80_MIN <= p.visual_authenticity.aspect_ratio <= ASPECT_RATIO_CR80_MAX) for p in pages)
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
            has_front_face_or_portrait=has_portrait,
            has_secure_qr=has_secure_qr,
            qr_position=qr_position,
            has_official_color_palette=has_palette,
            has_valid_card_geometry=has_geometry,
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
