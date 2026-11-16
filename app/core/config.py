from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Essential deployment and operational settings for the Aadhaar validator service."""

    app_name: str = "Aadhaar Validation Service"
    max_upload_mb: int = 12
    max_pdf_pages: int = 5
    pdf_render_dpi: int = 150
    models_dir: str = "models"
    ocr_det_model: str = "PP-OCRv5_mobile_det"
    ocr_rec_model: str = "devanagari_PP-OCRv5_mobile_rec"
    ocr_pool_size: int = 2
    ocr_threads: int = 6

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
