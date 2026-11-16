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

    def analyze_visual_authenticity(self, image: np.ndarray) -> dict:
        """
        Validates whether an image represents an authentic physical/digital Aadhaar card
        rather than plain typed text or empty paper.
        Evaluates aspect ratio (CR80/ID-1), 2D QR code presence, Aadhaar color palette
        (saffron, green, UIDAI maroon), and portrait/vibrancy density.
        """
        h, w = image.shape[:2]
        aspect_ratio = round(float(w / h) if h > 0 else 0.0, 3)
        total_pixels = max(1, w * h)

        # 1. Aspect Ratio: Standard Aadhaar card ratio is ~1.586 (tolerance 1.20 to 1.90)
        is_card_aspect = 1.20 <= aspect_ratio <= 1.90

        # 2. QR Code Pattern: Back side or e-Aadhaar contains high-density 2D QR code
        qr_detected = False
        try:
            detector = cv2.QRCodeDetector()
            ret, points = detector.detect(image)
            if ret and points is not None and len(points) > 0:
                pts = points[0] if len(points.shape) == 3 else points
                if len(pts) == 4:
                    area = cv2.contourArea(pts.astype(np.float32))
                    # Genuine card QR occupies between 2% and 45% of the card area
                    if (0.02 * total_pixels) <= area <= (0.45 * total_pixels):
                        d01 = np.linalg.norm(pts[0] - pts[1])
                        d12 = np.linalg.norm(pts[1] - pts[2])
                        if d01 > 0 and d12 > 0:
                            ratio = max(d01, d12) / min(d01, d12)
                            if ratio <= 1.35:  # must be approximately square
                                qr_detected = True
        except Exception:
            qr_detected = False

        # 3. Aadhaar Palette: Saffron/Orange top stripe, Green bottom, UIDAI Red/Maroon header
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None
        has_aadhaar_palette = False
        color_vibrancy_pct = 0.0

        if hsv is not None:
            saffron = cv2.inRange(hsv, np.array([5, 50, 100]), np.array([25, 255, 255]))
            green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            red1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
            red2 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
            red = cv2.bitwise_or(red1, red2)

            saffron_pct = round(float(np.count_nonzero(saffron)) / total_pixels * 100, 2)
            green_pct = round(float(np.count_nonzero(green)) / total_pixels * 100, 2)
            red_pct = round(float(np.count_nonzero(red)) / total_pixels * 100, 2)
            sat = hsv[:, :, 1]
            color_vibrancy_pct = round(float(np.count_nonzero(sat > 40)) / total_pixels * 100, 2)

            has_aadhaar_palette = (saffron_pct >= 0.5 or red_pct >= 0.5) and (green_pct >= 0.3 or red_pct >= 0.5)

        # 4. Portrait / Graphics presence (non-text graphical blocks)
        has_photo_or_graphics = color_vibrancy_pct > 3.0

        # Authenticity scoring
        checks = [
            (is_card_aspect, 20, "VALID_ASPECT_RATIO"),
            (has_aadhaar_palette, 25, "AADHAAR_PALETTE_DETECTED"),
            (has_photo_or_graphics, 25, "GRAPHICS_OR_PORTRAIT_REGION_DETECTED"),
            (qr_detected, 30, "SECURE_QR_PATTERN_DETECTED"),
        ]
        score = sum(pts for cond, pts, _ in checks if cond)
        checks_passed = [flag for cond, _, flag in checks if cond]

        return {
            "is_valid_aadhaar_card_image": score >= 50,
            "authenticity_score": score,
            "aspect_ratio": aspect_ratio,
            "qr_detected": qr_detected,
            "has_aadhaar_palette": has_aadhaar_palette,
            "has_photo_or_graphics": has_photo_or_graphics,
            "checks_passed": checks_passed,
        }

    def prepare_page_image(self, raw_image: np.ndarray) -> tuple[dict, np.ndarray]:
        """Single-pass quality diagnostics, dimension normalization, and card ROI isolation."""
        quality = self.analyze_quality(raw_image)
        normalized = self.normalize(raw_image)
        roi = self.isolate_card_roi(normalized)
        return quality, roi
