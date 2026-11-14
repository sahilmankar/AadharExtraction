from dependency_injector import containers, providers

from app.services.extractor_service import AadhaarExtractorService
from app.services.media_service import MediaService
from app.services.ocr_service import OCRService
from app.services.processing_service import DocumentProcessingService


class Container(containers.DeclarativeContainer):
    """IoC Dependency Injection Container managing singleton lifecycles."""

    wiring_config = containers.WiringConfiguration(modules=["app.api.document_routes"])

    config = providers.Configuration()

    # Unified Media Service (File read, OpenCV decode, PyMuPDF rasterize, quality check, ROI crop)
    media_service = providers.Singleton(
        MediaService,
        max_upload_bytes=config.max_upload_bytes,
        max_image_pixels=config.max_image_pixels,
        max_pdf_pages=config.max_pdf_pages,
        pdf_render_dpi=config.pdf_render_dpi,
    )
    file_service = media_service
    image_service = media_service

    # Multi-worker OCR Engine Pool singleton
    ocr_service = providers.Singleton(
        OCRService,
        language=config.ocr_lang,
        device=config.ocr_device,
        models_dir=config.models_dir,
        det_model=config.ocr_det_model,
        rec_model=config.ocr_rec_model,
        pool_size=config.ocr_pool_size,
        threads=config.ocr_threads,
    )

    extractor_service = providers.Singleton(AadhaarExtractorService)

    # Document Orchestrator Factory
    document_processor = providers.Factory(
        DocumentProcessingService,
        media_service=media_service,
        ocr_service=ocr_service,
        extractor_service=extractor_service,
    )
