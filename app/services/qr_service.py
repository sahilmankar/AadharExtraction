import logging
import re
import xml.etree.ElementTree as ET
import zlib
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UIDAI Secure QR Specification Constants
# ---------------------------------------------------------------------------
# Byte delimiter (0xFF = 255) separating text fields in decompressed V2/V3 QR payload
UIDAI_FIELD_DELIMITER = b"\xff"

# Standard minimum number of delimited fields in a valid UIDAI Secure QR payload
UIDAI_MIN_FIELDS_COUNT = 16

# Minimum fields count when biometric face JPEG thumbnail is embedded (index 20+)
UIDAI_PHOTO_FIELDS_THRESHOLD = 20

# Standard UIDAI Secure QR field positions after splitting on 0xFF:
# 1: Reference ID (last 4 digits are Aadhaar last-4)
# 2: Name
# 3: DOB (DD-MM-YYYY)
# 4: Gender (M/F)
# 5: Care of (C/O, S/O, W/O, D/O)
# 6: District
# 9: Locality / Street
# 10: PIN Code
# 12: State
# 15: Landmark
UIDAI_FIELD_REF_ID = 1
UIDAI_FIELD_NAME = 2
UIDAI_FIELD_DOB = 3
UIDAI_FIELD_GENDER = 4
UIDAI_FIELD_CARE_OF = 5
UIDAI_FIELD_DISTRICT = 6
UIDAI_FIELD_LOCALITY = 9
UIDAI_FIELD_PIN_CODE = 10
UIDAI_FIELD_STATE = 12
UIDAI_FIELD_LANDMARK = 15

# Address field indices in standard display order:
UIDAI_ADDRESS_FIELD_INDICES = [
    UIDAI_FIELD_CARE_OF,
    UIDAI_FIELD_LOCALITY,
    UIDAI_FIELD_LANDMARK,
    UIDAI_FIELD_DISTRICT,
    UIDAI_FIELD_STATE,
    UIDAI_FIELD_PIN_CODE,
]


def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-zA-Z]", "", name).lower()


def _fuzzy_name_match(name1: str | None, name2: str | None) -> bool:
    if not name1 or not name2:
        return False
    n1 = _normalize_name(name1)
    n2 = _normalize_name(name2)
    if n1 == n2:
        return True
    # Check if all tokens of one name exist in the other
    words1 = set(re.findall(r"[a-zA-Z]+", name1.lower()))
    words2 = set(re.findall(r"[a-zA-Z]+", name2.lower()))
    if words1 and words2 and (words1.issubset(words2) or words2.issubset(words1)):
        return True
    return False


def _normalize_dob(dob: str | None) -> str:
    if not dob:
        return ""
    # Standardize 17-11-2001 or 17/11/2001 to 17/11/2001
    return dob.replace("-", "/").strip()


def _normalize_gender(g: str | None) -> str:
    if not g:
        return ""
    g = g.strip().upper()
    if g.startswith("M") or "पुरुष" in g:
        return "M"
    if g.startswith("F") or "महिला" in g:
        return "F"
    if g.startswith("T") or "TRANS" in g:
        return "T"
    return g


class AadhaarQRService:
    """
    Decodes UIDAI Secure QR codes (V2/V3 compressed BigInteger) and legacy XML QR codes (V1),
    and performs cross-matching against OCR extracted demographic data.
    """

    def decode_qr_from_image(self, image: np.ndarray) -> dict | None:
        """Finds and decodes any UIDAI QR code in the image using pyzbar."""
        try:
            from pyzbar.pyzbar import decode
            decoded_objects = decode(image)
            for obj in decoded_objects:
                if obj.type == "QRCODE":
                    parsed = self.parse_qr_payload(obj.data)
                    if parsed:
                        return parsed
        except Exception as e:
            logger.debug("pyzbar QR decoding failed or not available: %s", e)

        return None

    def parse_qr_payload(self, raw_data: bytes) -> dict | None:
        """Parses raw QR payload: handles V2/V3 Secure BigInt Zlib binary and V1 XML."""
        if not raw_data:
            return None

        # 1. Try UIDAI Secure QR Code (V2 / V3: BigInteger decimal string or raw Deflate)
        secure_res = self._parse_secure_qr(raw_data)
        if secure_res:
            return secure_res

        # 2. Try UIDAI Legacy XML QR Code (V1)
        xml_res = self._parse_xml_qr(raw_data)
        if xml_res:
            return xml_res

        return None

    def _parse_secure_qr(self, raw_data: bytes) -> dict | None:
        try:
            decompressed: bytes
            if raw_data.isdigit():
                # UIDAI V2 payload encoded as BigInteger decimal string
                big_int = int(raw_data.decode("utf-8"))
                # Standard formula to calculate exact byte length needed to store big_int: ceil(bits / 8)
                byte_len = (big_int.bit_length() + 7) // 8
                raw_bytes = big_int.to_bytes(byte_len, byteorder="big")
                # 16 + zlib.MAX_WBITS enables gzip/deflate auto-detection (16 + 15 = 31)
                decompressed = zlib.decompress(raw_bytes, 16 + zlib.MAX_WBITS)
            else:
                decompressed = zlib.decompress(raw_data, 16 + zlib.MAX_WBITS)

            parts = decompressed.split(UIDAI_FIELD_DELIMITER)
            if len(parts) >= UIDAI_MIN_FIELDS_COUNT:
                def get_part(idx: int) -> str | None:
                    if idx < len(parts):
                        try:
                            val = parts[idx].decode("ISO-8859-1").strip()
                            return val if val else None
                        except Exception:
                            return None
                    return None

                ref_id = get_part(UIDAI_FIELD_REF_ID)
                aadhaar_last4 = ref_id[:4] if ref_id and len(ref_id) >= 4 and ref_id[:4].isdigit() else None
                dob = _normalize_dob(get_part(UIDAI_FIELD_DOB))
                gender_raw = get_part(UIDAI_FIELD_GENDER)
                gender = None
                if gender_raw:
                    g = gender_raw.upper()
                    gender = "MALE" if g.startswith("M") else ("FEMALE" if g.startswith("F") else g)

                address_items = [get_part(idx) for idx in UIDAI_ADDRESS_FIELD_INDICES if get_part(idx)]
                address = ", ".join(address_items) if address_items else None

                return {
                    "format": "SECURE_QR_V2",
                    "reference_id": ref_id,
                    "aadhaar_last4": aadhaar_last4,
                    "name": get_part(UIDAI_FIELD_NAME),
                    "dob": dob,
                    "gender": gender,
                    "care_of": get_part(UIDAI_FIELD_CARE_OF),
                    "district": get_part(UIDAI_FIELD_DISTRICT),
                    "pin_code": get_part(UIDAI_FIELD_PIN_CODE),
                    "state": get_part(UIDAI_FIELD_STATE),
                    "address": address,
                    "has_photo": len(parts) > UIDAI_PHOTO_FIELDS_THRESHOLD,
                }
        except Exception:
            pass
        return None

    def _parse_xml_qr(self, raw_data: bytes) -> dict | None:
        try:
            text = raw_data.decode("utf-8", errors="ignore")
            if "<PrintLetterBarcodeData" in text:
                root = ET.fromstring(text)
                attribs = root.attrib
                uid = attribs.get("uid", "")
                aadhaar_last4 = uid[-4:] if len(uid) >= 4 else None
                dob = _normalize_dob(attribs.get("dob") or attribs.get("yob"))
                gender_raw = attribs.get("gender")
                gender = None
                if gender_raw:
                    g = gender_raw.upper()
                    gender = "MALE" if g.startswith("M") else ("FEMALE" if g.startswith("F") else g)

                address_fields = [
                    attribs.get(k) for k in ["co", "house", "street", "lm", "loc", "vtc", "po", "dist", "state", "pc"]
                    if attribs.get(k)
                ]
                return {
                    "format": "XML_V1",
                    "reference_id": uid,
                    "aadhaar_last4": aadhaar_last4,
                    "name": attribs.get("name"),
                    "dob": dob,
                    "gender": gender,
                    "care_of": attribs.get("co"),
                    "district": attribs.get("dist"),
                    "pin_code": attribs.get("pc"),
                    "state": attribs.get("state"),
                    "address": ", ".join(address_fields) if address_fields else None,
                    "has_photo": False,
                }
        except Exception:
            pass
        return None

    def cross_match(self, ocr_data: Any, qr_data: dict | None, has_visual_qr: bool = False) -> dict:
        """
        Cross-validates OCR extracted data against decoded QR payload.
        Detects tampering or counterfeit documents.
        """
        if not qr_data:
            status = "PAYLOAD_UNREADABLE_LOW_RESOLUTION" if has_visual_qr else "QR_NOT_FOUND"
            return {
                "cross_match_status": status,
                "match_confidence": 0.0,
                "name_matched": None,
                "dob_matched": None,
                "gender_matched": None,
                "aadhaar_last4_matched": None,
                "pin_code_matched": None,
                "mismatches": [],
                "qr_data": None,
            }

        matched_flags: dict[str, bool | None] = {
            "name_matched": None,
            "dob_matched": None,
            "gender_matched": None,
            "aadhaar_last4_matched": None,
            "pin_code_matched": None,
        }
        checks: list[bool] = []
        mismatches: list[str] = []

        field_comparisons = [
            (
                "name_matched",
                getattr(ocr_data, "name", None),
                qr_data.get("name"),
                lambda o, q: _fuzzy_name_match(o, q),
                lambda o, q: f"NAME_MISMATCH: OCR '{o}' vs QR '{q}'",
            ),
            (
                "dob_matched",
                getattr(ocr_data, "dob", None),
                qr_data.get("dob"),
                lambda o, q: _normalize_dob(o) == _normalize_dob(q),
                lambda o, q: f"DOB_MISMATCH: OCR '{o}' vs QR '{q}'",
            ),
            (
                "gender_matched",
                getattr(ocr_data, "gender", None),
                qr_data.get("gender"),
                lambda o, q: _normalize_gender(o) == _normalize_gender(q),
                lambda o, q: f"GENDER_MISMATCH: OCR '{o}' vs QR '{q}'",
            ),
            (
                "aadhaar_last4_matched",
                re.sub(r"\D", "", getattr(ocr_data, "aadhaar_number", "") or ""),
                qr_data.get("aadhaar_last4"),
                lambda o, q: o.endswith(q),
                lambda o, q: f"AADHAAR_LAST4_MISMATCH: OCR ends with '{o[-4:]}' vs QR '{q}'",
            ),
            (
                "pin_code_matched",
                (getattr(ocr_data, "pin_code", None) or "").strip(),
                (qr_data.get("pin_code") or "").strip(),
                lambda o, q: o == q,
                lambda o, q: f"PIN_CODE_MISMATCH: OCR '{o}' vs QR '{q}'",
            ),
        ]

        for flag_name, ocr_val, qr_val, comparator, err_gen in field_comparisons:
            if ocr_val and qr_val:
                is_match = bool(comparator(ocr_val, qr_val))
                matched_flags[flag_name] = is_match
                checks.append(is_match)
                if not is_match:
                    mismatches.append(err_gen(ocr_val, qr_val))

        if not checks:
            status = "MATCHED"
            confidence = 0.5
        elif all(checks):
            status = "MATCHED"
            confidence = 1.0
        else:
            status = "MISMATCH_DETECTED"
            confidence = round(sum(checks) / len(checks), 2)

        return {
            "cross_match_status": status,
            "match_confidence": confidence,
            "mismatches": mismatches,
            "qr_data": qr_data,
            **matched_flags,
        }
