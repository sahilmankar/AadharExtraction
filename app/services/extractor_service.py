import re
from typing import Literal
from app.schemas.document import AadhaarExtractedData, OCRLine

VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

DOB_PATTERNS = [
    r"(?:DOB|Date of Birth|जन्म\s*तारीख|जन्म\s*तिथि)[\s\:\-]+([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})",
    r"\b([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b",
    r"(?:Year of Birth|जन्म\s*वर्ष)[\s\:\-]+([12][0-9]{3})",
]

HEADER_KEYWORDS = (
    "government", "india", "authority", "unique", "identification",
    "aadhaar", "आधार", "भारत", "सरकार", "मेरा आधार", "मेरी पहचान",
)


def validate_verhoeff(number_str: str) -> bool:
    """Validates 12-digit Aadhaar number using standard Verhoeff checksum algorithm."""
    clean = re.sub(r"\D", "", number_str)
    if len(clean) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed([int(d) for d in clean])):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][item]]
    return c == 0


class AadhaarExtractorService:
    """Extracts, normalizes, and merges demographic identity attributes from OCR lines."""

    def extract_from_lines(self, lines: list[OCRLine], full_text: str) -> AadhaarExtractedData:
        data = AadhaarExtractedData()

        # 1. Aadhaar Number (Masked or Unmasked)
        if masked := re.search(r"\b([X\*\.]{4}[\s\-]?[X\*\.]{4}[\s\-]?[0-9]{4})\b", full_text, re.IGNORECASE):
            clean = re.sub(r"[\s\-]", "", masked.group(1).upper())
            data.aadhaar_number = f"{clean[:4]} {clean[4:8]} {clean[8:12]}"
            data.is_masked, data.is_aadhaar_number_valid = True, True
        elif unmasked := re.search(r"\b([2-9][0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4})\b", full_text):
            clean = re.sub(r"[\s\-]", "", unmasked.group(1))
            data.aadhaar_number = f"{clean[:4]} {clean[4:8]} {clean[8:12]}"
            data.is_masked = False
            data.is_aadhaar_number_valid = validate_verhoeff(clean)

        # 2. Date of Birth or Year of Birth
        for pattern in DOB_PATTERNS:
            if dob_match := re.search(pattern, full_text, re.IGNORECASE):
                data.dob = dob_match.group(1)
                break

        # 3. Gender
        if g_match := re.search(r"\b(MALE|FEMALE|TRANSGENDER|पुरुष|महिला)\b", full_text, re.IGNORECASE):
            val = g_match.group(1).upper()
            data.gender = "MALE" if val in ["पुरुष", "MALE"] else ("FEMALE" if val in ["महिला", "FEMALE"] else "TRANSGENDER")

        # 4. PIN Code
        if pin_matches := re.findall(r"\b([1-9][0-9]{5})\b", full_text):
            data.pin_code = pin_matches[-1]

        # 5. Name heuristics (scan lines before DOB)
        text_lines = [line.text.strip() for line in lines if line.text.strip()]
        dob_idx = next((i for i, t in enumerate(text_lines) if re.search(r"(?:DOB|जन्म|Date of Birth|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})", t, re.IGNORECASE)), -1)
        if dob_idx > 0:
            candidates = []
            for i in range(max(0, dob_idx - 3), dob_idx):
                cand = text_lines[i]
                if any(kw in cand.lower() for kw in HEADER_KEYWORDS):
                    continue
                clean_cand = re.sub(r"[^A-Za-z\s\.\-]", "", cand).strip()
                if len(clean_cand) >= 3 and not re.search(r"\b(DOB|MALE|FEMALE|YEAR)\b", clean_cand, re.IGNORECASE):
                    candidates.append(clean_cand)
            if candidates:
                data.name = candidates[-1]

        # 6. Address heuristics
        address_markers = ("address:", "पता:", "address", "पता", "c/o", "s/o", "w/o", "d/o")
        for idx, t in enumerate(text_lines):
            if any(t.lower().startswith(m) or m in t.lower() for m in address_markers):
                parts = [t]
                for nxt in range(idx + 1, min(idx + 5, len(text_lines))):
                    nxt_line = text_lines[nxt]
                    if re.search(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b", nxt_line):
                        break
                    parts.append(nxt_line)
                    if data.pin_code and data.pin_code in nxt_line:
                        break
                data.address = ", ".join(parts)
                break

        # 7. Document side detection
        has_front = bool(data.dob or data.gender or re.search(r"\b(dob|male|female|जन्म)\b", full_text, re.IGNORECASE))
        has_back = bool(data.address or re.search(r"\b(address|पता|help@uidai|1947)\b", full_text, re.IGNORECASE))
        data.document_side = "FULL_PAGE" if (has_front and has_back) else ("FRONT" if has_front else ("BACK" if has_back else "UNKNOWN"))

        return data

    def merge_extracted_data(self, data_list: list[AadhaarExtractedData]) -> AadhaarExtractedData:
        """Merges extracted demographic attributes across pages or dual-side scans."""
        merged = AadhaarExtractedData()
        for d in data_list:
            if not merged.aadhaar_number and d.aadhaar_number:
                merged.aadhaar_number, merged.is_aadhaar_number_valid, merged.is_masked = d.aadhaar_number, d.is_aadhaar_number_valid, d.is_masked
            elif merged.aadhaar_number and merged.is_masked and not d.is_masked and d.aadhaar_number:
                merged.aadhaar_number, merged.is_aadhaar_number_valid, merged.is_masked = d.aadhaar_number, d.is_aadhaar_number_valid, False

            for field in ("name", "dob", "gender", "address", "pin_code"):
                if not getattr(merged, field) and (val := getattr(d, field)):
                    setattr(merged, field, val)

        sides = {d.document_side for d in data_list}
        merged.document_side = "FULL_PAGE" if ("FULL_PAGE" in sides or ("FRONT" in sides and "BACK" in sides)) else ("FRONT" if "FRONT" in sides else ("BACK" if "BACK" in sides else "UNKNOWN"))
        return merged

    def classify_document(self, merged: AadhaarExtractedData, full_text: str) -> Literal["AADHAAR_FRONT", "AADHAAR_BACK", "E_AADHAAR", "NOT_CLASSIFIED"]:
        """Classifies document layout based on extracted attributes and keyword semantics."""
        has_kw = bool(
            re.search(r"(aadhaar|आधार|uidai|unique identification authority|भारत सरकार|government of india)", full_text, re.IGNORECASE)
            or merged.aadhaar_number
        )
        if not has_kw:
            return "NOT_CLASSIFIED"

        if merged.document_side == "FULL_PAGE" or (merged.dob and merged.address):
            return "E_AADHAAR"
        if merged.document_side == "FRONT" or (merged.dob and not merged.address):
            return "AADHAAR_FRONT"
        if merged.document_side == "BACK" or (merged.address and not merged.dob):
            return "AADHAAR_BACK"
        return "NOT_CLASSIFIED"
