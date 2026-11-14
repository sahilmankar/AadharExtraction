import cv2
import fitz
import numpy as np
from fastapi import UploadFile


class InvalidDocumentError(Exception):
    """Raised when an uploaded document is corrupt, empty, unsupported, or exceeds limits."""
    pass


class MediaService:
    """
    Unified service handling document byte ingestion, image decoding,
    PDF rasterization, image quality diagnostics, and ROI isolation.
    """

    def __init__(
        self,
        max_upload_bytes: int = 12 * 1024 * 1024,
        max_image_pixels: int = 20_000_000,
        max_pdf_pages: int = 5,
        pdf_render_dpi: int = 150,
    ):
        self.max_upload_bytes = max_upload_bytes
        self.max_image_pixels = max_image_pixels
        self.max_pdf_pages = max_pdf_pages
        self.pdf_render_dpi = pdf_render_dpi

    async def read_upload(self, upload: UploadFile) -> bytes:
        data = await upload.read(self.max_upload_bytes + 1)
        if not data:
            raise InvalidDocumentError("Uploaded file is empty.")
        if len(data) > self.max_upload_bytes:
            raise InvalidDocumentError(
                f"File size exceeds maximum limit of {self.max_upload_bytes // (1024 * 1024)}MB."
            )
        return data

    def decode_image(self, data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidDocumentError("Unable to decode image. Ensure file is a valid JPG or PNG.")
        h, w = image.shape[:2]
        if h * w > self.max_image_pixels:
            raise InvalidDocumentError(f"Image pixel count ({h * w}) exceeds limit ({self.max_image_pixels}).")
        return image

    def pdf_to_images(self, data: bytes) -> list[np.ndarray]:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise InvalidDocumentError(f"Could not parse PDF file: {e}")

        if doc.is_encrypted or doc.needs_pass:
            raise InvalidDocumentError("PDF is password protected. Please upload an unlocked PDF.")
        if len(doc) == 0:
            raise InvalidDocumentError("PDF contains no pages.")
        if len(doc) > self.max_pdf_pages:
            raise InvalidDocumentError(f"PDF page count ({len(doc)}) exceeds limit of {self.max_pdf_pages}.")

        images = []
        zoom = self.pdf_render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
            images.append(img)
        doc.close()
        return images

    def analyze_quality(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        warnings = []
        if w < 600 or h < 400:
            warnings.append("LOW_RESOLUTION")
        if blur_score < 50.0:
            warnings.append("POSSIBLY_BLURRY")
        if brightness < 50.0:
            warnings.append("TOO_DARK")
        if brightness > 220.0:
            warnings.append("TOO_BRIGHT")

        return {
            "width": w,
            "height": h,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "status": "WARNING" if warnings else "GOOD",
            "warnings": warnings,
        }

    analyze = analyze_quality  # Alias for backward compatibility

    def normalize(self, image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image

    def isolate_card_roi(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # Portrait A4 e-Aadhaar: lower 45% holds the card cutouts
        if h > 1.25 * w:
            return image[int(h * 0.55):h, 0:w]

        # Camera/scanner background contour detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_area = h * w
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if 0.15 * total_area < area < 0.96 * total_area and ch > 0:
                    if 1.2 <= (cw / ch) <= 1.9:
                        px, py = int(cw * 0.02), int(ch * 0.02)
                        return image[max(0, y - py):min(h, y + ch + py), max(0, x - px):min(w, x + cw + px)]
        except Exception:
            pass
        return image

    def prepare_page_image(self, raw_image: np.ndarray) -> tuple[dict, np.ndarray]:
        """Single-pass quality diagnostics, dimension normalization, and card ROI isolation."""
        quality = self.analyze_quality(raw_image)
        normalized = self.normalize(raw_image)
        roi = self.isolate_card_roi(normalized)
        return quality, roi
