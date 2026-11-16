import logging
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.api.document_routes import router
from app.core.config import get_settings
from app.core.container import Container
from app.services.media_service import InvalidDocumentError

# Direct PaddleX to project model cache and enable multi-core BLAS
models_path = str(Path(__file__).resolve().parent.parent / "models")
os.environ["PADDLE_PDX_CACHE_HOME"] = models_path
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("aadhaar_validator")
settings = get_settings()

# Initialize and configure Dependency Injection Container
container = Container()
container.config.from_dict({
    "app_name": settings.app_name,
    "max_upload_bytes": settings.max_upload_bytes,
    "max_pdf_pages": settings.max_pdf_pages,
    "pdf_render_dpi": settings.pdf_render_dpi,
    "models_dir": settings.models_dir,
    "ocr_det_model": settings.ocr_det_model,
    "ocr_rec_model": settings.ocr_rec_model,
    "ocr_pool_size": settings.ocr_pool_size,
    "ocr_threads": settings.ocr_threads,
})


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing %s OCR engine worker pool...", settings.app_name)
    await run_in_threadpool(container.ocr_service)
    logger.info("OCR worker pool ready.")
    yield
    container.unwire()


app = FastAPI(
    title=settings.app_name,
    description="Privacy-focused local microservice for Aadhaar document parsing, local OCR, and UIDAI Verhoeff validation.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"tryItOutEnabled": True, "filter": True},
)
app.container = container


@app.exception_handler(InvalidDocumentError)
async def invalid_doc_handler(request: Request, exc: InvalidDocumentError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{duration:.4f}s"
        logger.info("HTTP %s %s completed in %.3fs [%d]", request.method, request.url.path, duration, response.status_code)
        return response
    except Exception as e:
        logger.error("HTTP %s %s failed after %.3fs: %s", request.method, request.url.path, time.perf_counter() - start, e)
        raise e


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"], summary="Health Check")
async def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "ocr_det_model": settings.ocr_det_model,
        "ocr_rec_model": settings.ocr_rec_model,
        "ocr_pool_size": settings.ocr_pool_size,
    }


app.include_router(router)
