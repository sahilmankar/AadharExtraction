import cv2
import fitz
import numpy as np
from fastapi import UploadFile

# ---------------------------------------------------------------------------
# Physical & Algorithmic Calibration Constants
# ---------------------------------------------------------------------------
# Standard typographic PDF resolution (1 inch = 72 PostScript points)
PDF_BASE_POINTS_PER_INCH = 72.0

# ISO/IEC 7810 ID-1 standard dimensions: 85.60 mm x 53.98 mm -> Ratio ~ 1.586
CARD_CR80_ASPECT_RATIO = 1.586
CARD_ASPECT_MIN = 1.15  # Accommodates camera perspective tilt and margin variance
CARD_ASPECT_MAX = 2.15

# Edge safety padding (2% of card dimensions) around isolated bounding box
CARD_BORDER_PADDING_RATIO = 0.02

# QR Code geometry: intrinsically square (1.0). Tolerances cover perspective distortion
QR_ASPECT_MIN = 0.78
QR_ASPECT_MAX = 1.28

# QR Code area bounds: occupies 1.5% to 35% of total card surface area
QR_AREA_MIN_RATIO = 0.015
QR_AREA_MAX_RATIO = 0.35

# High-frequency Laplacian variance threshold: 2D QR matrix contains dense dot transitions (>180)
QR_TEXTURE_MIN_VARIANCE = 180.0

# Image clarity diagnostics
QUALITY_BLUR_MIN_VAR = 50.0         # Laplacian variance < 50 indicates motion or focus blur
QUALITY_BRIGHTNESS_MIN = 50.0       # Mean grayscale < 50 indicates underexposure (too dark)
QUALITY_BRIGHTNESS_MAX = 220.0      # Mean grayscale > 220 indicates overexposure (washed out)
MIN_LEGIBLE_WIDTH = 600             # Minimum pixel width for reliable character segmentation
MIN_LEGIBLE_HEIGHT = 400
MAX_NORMALIZED_DIMENSION = 2000     # Downscale upper bound to cap OCR latency and RAM usage


class InvalidDocumentError(Exception):
    """Raised when an uploaded document is corrupt, empty, unsupported, or exceeds limits."""
    pass


def _detect_qr_region(image: np.ndarray, total_pixels: int) -> bool:
    """
    Detects 2D barcode presence using OpenCV finder pattern detection
    with high-frequency gradient/texture density fallback for compressed scans.
    """
    # 1. Primary: Standard OpenCV concentric finder pattern detection
    try:
        detector = cv2.QRCodeDetector()
        ret, points = detector.detect(image)
        if ret and points is not None and len(points) > 0:
            pts = points[0] if len(points.shape) == 3 else points
            if len(pts) == 4:
                area = cv2.contourArea(pts.astype(np.float32))
                if (QR_AREA_MIN_RATIO * total_pixels) <= area <= (0.45 * total_pixels):
                    d01 = np.linalg.norm(pts[0] - pts[1])
                    d12 = np.linalg.norm(pts[1] - pts[2])
                    if d01 > 0 and d12 > 0 and (max(d01, d12) / min(d01, d12)) <= 1.35:
                        return True
    except Exception:
        pass

    # 2. Secondary fallback: Dense 2D matrix texture detection (Sobel magnitude + morphological close)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 11x11 kernel merges alternating QR dots into a solid square contour
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(norm_mag, cv2.MORPH_CLOSE, kernel)
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5x5 open filter strips away thin linear noise (text lines, table borders)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if (QR_AREA_MIN_RATIO * total_pixels) <= area <= (QR_AREA_MAX_RATIO * total_pixels) and ch > 0:
                ar = float(cw) / float(ch)
                if QR_ASPECT_MIN <= ar <= QR_ASPECT_MAX:
                    roi_gray = gray[y:y+ch, x:x+cw]
                    if cv2.Laplacian(roi_gray, cv2.CV_64F).var() > QR_TEXTURE_MIN_VARIANCE:
                        return True
    except Exception:
        pass

    return False


def _check_aadhaar_palette(hsv: np.ndarray, total_pixels: int) -> tuple[bool, float]:
    """Evaluates presence of official Saffron, Green, and UIDAI Emblem Red hues in HSV space."""
    # Saffron stripe: Hue 5-25 deg
    saffron = cv2.inRange(hsv, np.array([5, 50, 100]), np.array([25, 255, 255]))
    # Green stripe: Hue 35-85 deg
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    # UIDAI Maroon/Emblem Red: Hue 0-10 deg or 170-180 deg
    red1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    saffron_pct = round(float(np.count_nonzero(saffron)) / total_pixels * 100, 2)
    green_pct = round(float(np.count_nonzero(green)) / total_pixels * 100, 2)
    red_pct = round(float(np.count_nonzero(red)) / total_pixels * 100, 2)

    # Color saturation > 40 indicates photographic portrait/graphics presence
    sat = hsv[:, :, 1]
    color_vibrancy_pct = round(float(np.count_nonzero(sat > 40)) / total_pixels * 100, 2)

    has_palette = (saffron_pct >= 0.5 or red_pct >= 0.5) and (green_pct >= 0.3 or red_pct >= 0.5)
    return has_palette, color_vibrancy_pct


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
        zoom = self.pdf_render_dpi / PDF_BASE_POINTS_PER_INCH
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            # For scanned image PDFs without digital text layers, extract embedded scan directly
            # to preserve lossless resolution and avoid rasterization blur on dense QR codes
            img_list = page.get_images()
            if len(page.get_text().strip()) == 0 and len(img_list) >= 1:
                try:
                    best_img, max_area = None, 0
                    for img_info in img_list:
                        base = doc.extract_image(img_info[0])
                        w_i, h_i = base.get("width", 0), base.get("height", 0)
                        if w_i * h_i > max_area and w_i >= 250 and h_i >= 180:
                            max_area = w_i * h_i
                            best_img = base
                    if best_img:
                        decoded = cv2.imdecode(np.frombuffer(best_img["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
                        if decoded is not None and decoded.size > 0:
                            images.append(decoded)
                            continue
                except Exception:
                    pass

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
        if w < MIN_LEGIBLE_WIDTH or h < MIN_LEGIBLE_HEIGHT:
            warnings.append("LOW_RESOLUTION")
        if blur_score < QUALITY_BLUR_MIN_VAR:
            warnings.append("POSSIBLY_BLURRY")
        if brightness < QUALITY_BRIGHTNESS_MIN:
            warnings.append("TOO_DARK")
        if brightness > QUALITY_BRIGHTNESS_MAX:
            warnings.append("TOO_BRIGHT")

        return {
            "width": w,
            "height": h,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "status": "WARNING" if warnings else "GOOD",
            "warnings": warnings,
        }

    def normalize(self, image: np.ndarray, max_dim: int = MAX_NORMALIZED_DIMENSION) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image

    def isolate_card_roi(self, image: np.ndarray) -> np.ndarray:
        """Dynamically locates and crops the card boundary anywhere on portrait or landscape canvases."""
        h, w = image.shape[:2]
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_area = h * w
            candidates = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                # Card contour must cover 6% to 94% of page area with aspect ratio between 1.15 and 2.15
                if (0.06 * total_area) < area < (0.94 * total_area) and ch > 0:
                    ar = cw / ch
                    if CARD_ASPECT_MIN <= ar <= CARD_ASPECT_MAX:
                        candidates.append((area, x, y, cw, ch))
            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                _, x, y, cw, ch = candidates[0]
                px, py = int(cw * CARD_BORDER_PADDING_RATIO), int(ch * CARD_BORDER_PADDING_RATIO)
                return image[max(0, y - py):min(h, y + ch + py), max(0, x - px):min(w, x + cw + px)]
        except Exception:
            pass

        return image

    def analyze_visual_authenticity(self, image: np.ndarray) -> dict:
        """Evaluates card format: CR80 aspect ratio, QR presence, official color palette, and portrait."""
        h, w = image.shape[:2]
        aspect_ratio = round(float(w / h) if h > 0 else 0.0, 3)
        total_pixels = max(1, w * h)

        # 1. Aspect Ratio: Standard CR80 card ratio ~1.586 (tolerance 1.20 to 1.90)
        is_card_aspect = 1.20 <= aspect_ratio <= 1.90

        # 2. QR Code Pattern: Evaluates finder patterns & 2D matrix texture
        qr_detected = _detect_qr_region(image, total_pixels)

        # 3. Aadhaar Palette: Saffron/Orange top stripe, Green bottom, UIDAI Red/Maroon header
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None
        has_aadhaar_palette, color_vibrancy_pct = _check_aadhaar_palette(hsv, total_pixels) if hsv is not None else (False, 0.0)

        # 4. Portrait / Graphics presence (non-text graphical blocks with color saturation > 40)
        has_photo_or_graphics = color_vibrancy_pct > 3.0

        # Authenticity scoring: weights reflect physical security elements
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
