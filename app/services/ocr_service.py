import logging
import os
import queue
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def _to_bbox_rect(box: Any) -> list[int] | None:
    """Converts 4-point polygon or flat coordinate array into [xmin, ymin, xmax, ymax]."""
    try:
        arr = np.array(box)
        if arr.ndim == 2 and len(arr) >= 4:
            return [int(np.min(arr[:, 0])), int(np.min(arr[:, 1])), int(np.max(arr[:, 0])), int(np.max(arr[:, 1]))]
        elif arr.ndim == 1 and len(arr) == 4:
            return [int(x) for x in arr]
    except Exception:
        pass
    return None


class OCRService:
    """Multi-core OCR worker pool using PaddleOCR 3.x with project-local models."""

    def __init__(
        self,
        language: str = "hi",
        device: str = "cpu",
        models_dir: str = "models",
        det_model: str = "PP-OCRv5_mobile_det",
        rec_model: str = "devanagari_PP-OCRv5_mobile_rec",
        pool_size: int = 2,
        threads: int = 4,
    ):
        self.language = language
        self.device = device
        self.models_dir = os.path.abspath(models_dir)
        self.det_model = det_model
        self.rec_model = rec_model
        self.pool_size = pool_size

        os.environ["PADDLE_PDX_CACHE_HOME"] = self.models_dir
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

        self._pool = queue.Queue(maxsize=pool_size)
        self._init_pool()

    def _create_engine(self):
        from paddleocr import PaddleOCR
        return PaddleOCR(
            lang=self.language,
            device=self.device,
            text_detection_model_name=self.det_model,
            text_recognition_model_name=self.rec_model,
            text_det_limit_side_len=960,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
        )

    def _init_pool(self):
        logger.info("Initializing OCR engine pool (size=%d, models_dir=%s)...", self.pool_size, self.models_dir)
        for _ in range(self.pool_size):
            self._pool.put(self._create_engine())
        logger.info("OCR engine pool ready.")

    def extract(self, image: np.ndarray) -> dict:
        """Borrows an engine worker from the pool, runs OCR, and returns it."""
        engine = self._pool.get()
        try:
            results = engine.predict(input=image)
            lines = []
            for res in results:
                payload = getattr(res, "json", res)
                data = payload.get("res", payload) if isinstance(payload, dict) else payload
                if isinstance(data, dict):
                    texts = data.get("rec_texts", [])
                    scores = data.get("rec_scores", [])
                    boxes = data.get("dt_polys", data.get("rec_boxes", []))
                    for i, text in enumerate(texts):
                        clean_text = str(text).strip()
                        if clean_text:
                            conf = float(scores[i]) if i < len(scores) else 0.9
                            box = boxes[i] if i < len(boxes) else None
                            lines.append({
                                "text": clean_text,
                                "confidence": round(conf, 4),
                                "bbox": _to_bbox_rect(box),
                            })

            if not lines:
                return {"status": "NO_TEXT", "full_text": "", "lines": []}

            return {
                "status": "SUCCESS",
                "full_text": " ".join(line["text"] for line in lines),
                "lines": lines,
            }
        finally:
            self._pool.put(engine)
