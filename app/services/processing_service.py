import asyncio
import logging
import time
import numpy as np
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.interfaces import IAadhaarExtractor, IDocumentProcessingService, IMediaService, IOCREngine
from app.schemas.document import DocumentPageResult, DocumentProcessingResponse, ImageQualityResult, OCRLine, OCRResult

logger = logging.getLogger(__name__)


class DocumentProcessingService(IDocumentProcessingService):
    """
    Concurrent document processing orchestrator.
    Handles parallel multi-page scanning, ROI isolation, OCR, and demographic aggregation.
    """

    def __init__(
        self,
        media_service: IMediaService,
        ocr_service: IOCREngine,
        extractor_service: IAadhaarExtractor,
        file_service: IMediaService | None = None,
        image_service: IMediaService | None = None,
    ):
        self.media_service = media_service or file_service or image_service
        self.ocr_service = ocr_service
        self.extractor_service = extractor_service

    async def _process_single_page(self, page_num: int, raw_image: np.ndarray) -> DocumentPageResult:
        t_start = time.perf_counter()

        # 1. Single-pass quality analysis, normalization & ROI isolation
        t0 = time.perf_counter()
        quality_dict, roi_img = await run_in_threadpool(self.media_service.prepare_page_image, raw_image)
        t_prep = time.perf_counter() - t0

        # 2. OCR inference on isolated ROI
        t0 = time.perf_counter()
        ocr_data = await run_in_threadpool(self.ocr_service.extract, roi_img)
        lines = [OCRLine(**line) for line in ocr_data["lines"]]
        ocr_result = OCRResult(status=ocr_data["status"], full_text=ocr_data["full_text"], lines=lines)
        t_ocr = time.perf_counter() - t0

        # 3. Demographic extraction
        t0 = time.perf_counter()
        page_extracted = self.extractor_service.extract_from_lines(lines, ocr_data["full_text"])
        t_ext = time.perf_counter() - t0

        logger.info(
            "Page %d processed in %.3fs (Prep+ROI: %.3fs, OCR: %.3fs, Extraction: %.3fs)",
            page_num, time.perf_counter() - t_start, t_prep, t_ocr, t_ext,
        )

        return DocumentPageResult(
            page_number=page_num,
            image_quality=ImageQualityResult(**quality_dict),
            ocr=ocr_result,
            extracted_data=page_extracted,
        )

    async def process_images(self, images: list[np.ndarray]) -> DocumentProcessingResponse:
        t_start = time.perf_counter()

        # Concurrently process all pages across worker pool
        pages = await asyncio.gather(*[self._process_single_page(i, img) for i, img in enumerate(images, start=1)])
        pages.sort(key=lambda p: p.page_number)

        # Merge fields across pages and classify side
        extracted_list = [p.extracted_data for p in pages if p.extracted_data is not None]
        aggregated_data = self.extractor_service.merge_extracted_data(extracted_list)
        combined_text = " ".join(p.ocr.full_text for p in pages)
        doc_type = self.extractor_service.classify_document(aggregated_data, combined_text)

        status = "SUCCESS" if any(p.ocr.status == "SUCCESS" for p in pages) else "PARTIAL"
        duration = time.perf_counter() - t_start

        logger.info("Processed %d page(s) concurrently in %.3fs", len(pages), duration)

        return DocumentProcessingResponse(
            processing_status=status,
            document_type=doc_type,
            authenticity_status="NOT_VERIFIED",
            pages=pages,
            aggregated_data=aggregated_data,
            message="Image processing and OCR completed successfully.",
            processing_time_seconds=round(duration, 3),
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
