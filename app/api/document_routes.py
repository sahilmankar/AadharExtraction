import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
import numpy as np
from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.schemas.document import DocumentProcessingResponse
from app.services.processing_service import DocumentProcessingService

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post("/images", response_model=DocumentProcessingResponse, summary="Extract Aadhaar Front/Back Images")
@inject
async def upload_images(
    front_image: UploadFile = File(..., description="Front Aadhaar card image (JPG/PNG)"),
    back_image: UploadFile | None = File(None, description="Optional Back Aadhaar card image (JPG/PNG)"),
    processor: DocumentProcessingService = Depends(Provide[Container.document_processor]),
):
    return await processor.process_image_upload(front_image, back_image)


@router.post("/pdf", response_model=DocumentProcessingResponse, summary="Extract Multi-Page PDF e-Aadhaar")
@inject
async def upload_pdf(
    file: UploadFile = File(..., description="Unlocked Aadhaar PDF document"),
    processor: DocumentProcessingService = Depends(Provide[Container.document_processor]),
):
    return await processor.process_pdf_upload(file)


@router.post("/test-synthetic", response_model=DocumentProcessingResponse, summary="[Swagger Test] 1-Click Synthetic Card Test")
@inject
async def test_synthetic_aadhaar(
    processor: DocumentProcessingService = Depends(Provide[Container.document_processor]),
):
    img = np.full((700, 1000, 3), 180, dtype=np.uint8)
    cv2.putText(img, "Government of India", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Rohan Sharma", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "DOB: 12/04/1995", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Male", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "9999 4105 7036", (80, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return await processor.process_images([img])


@router.get("/sample-test-image", summary="[Swagger Test] Download Sample Test Image")
async def download_sample_test_image():
    img = np.full((700, 1000, 3), 180, dtype=np.uint8)
    cv2.putText(img, "Government of India", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Rohan Sharma", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "DOB: 12/04/1995", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Male", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "9999 4105 7036", (80, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode test image")
    return Response(content=encoded.tobytes(), media_type="image/png")
