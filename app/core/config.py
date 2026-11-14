from functools import lru_cache
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Aadhaar Validation Service"
    app_env: str = "development"
    max_upload_mb: int = 12
    max_image_pixels: int = 20_000_000
    max_pdf_pages: int = 5
    pdf_render_dpi: int = 150  # 150 DPI is ideal balance between high OCR accuracy and speed
    ocr_lang: str = "hi"
    ocr_device: str = "cpu"
    models_dir: str = "models"
    ocr_det_model: str = "PP-OCRv5_mobile_det"
    ocr_rec_model: str = "devanagari_PP-OCRv5_mobile_rec"
    ocr_pool_size: int = 2
    ocr_threads: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
