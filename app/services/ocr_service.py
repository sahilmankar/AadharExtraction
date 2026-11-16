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
        det_limit: int = 960,
        rec_batch_size: int = 6,
        pool_size: int = 2,
        threads: int = 6,
    ):
        self.language = language
        self.device = device
        self.models_dir = os.path.abspath(models_dir)
        self.det_model = det_model
        self.rec_model = rec_model
        self.det_limit = det_limit
        self.rec_batch_size = rec_batch_size
        self.pool_size = pool_size

        os.environ["PADDLE_PDX_CACHE_HOME"] = self.models_dir
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

        self._pool = queue.Queue(maxsize=pool_size)
        self._init_pool()

    def _create_engine(self):
        from paddleocr import PaddleOCR
        kwargs: dict[str, Any] = {
            "device": self.device,
            "text_det_limit_side_len": self.det_limit,
            "text_recognition_batch_size": self.rec_batch_size,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": "paddle",
        }
        if self.det_model:
            kwargs["text_detection_model_name"] = self.det_model
        if self.rec_model:
            kwargs["text_recognition_model_name"] = self.rec_model
        if not self.det_model and not self.rec_model:
            kwargs["lang"] = self.language

        return PaddleOCR(**kwargs)

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
