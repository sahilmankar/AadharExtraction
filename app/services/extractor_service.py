import re
from typing import Literal
from app.schemas.document import AadhaarExtractedData, OCRLine

# ---------------------------------------------------------------------------
# Verhoeff Dihedral Group D_10 Checksum Tables
# ---------------------------------------------------------------------------
# D_5 multiplication table extended to base 10 (detects all single-digit errors & adjacent transpositions)
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

# Cyclic permutation table modulo 8 applied to each digit position
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

# ---------------------------------------------------------------------------
# Demographic Layout Heuristics Constants
# ---------------------------------------------------------------------------
MIN_NAME_LENGTH = 3         # Shortest valid English name length in India (e.g. "Ram", "Ali")
NAME_LOOKBACK_LINES = 3     # Resident name is printed 1 to 3 lines above DOB/Year of Birth
MAX_ADDRESS_LINES = 8       # Aadhaar address block spans at most 8 lines before UIDAI footer

DOB_PATTERNS = [
    r"(?:DOB|Date of Birth|जन्म\s*तारीख|जन्म\s*तिथि)[\s\:\-]+([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})",
    r"\b([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b",
    r"(?:Year of Birth|Year of Bith|जन्म\s*वर्ष|वर्ष)[^\d]*([12][0-9]{3})",
]

HEADER_KEYWORDS = (
    "government", "india", "authority", "unique", "identification",
    "aadhaar", "आधार", "भारत", "सरकार", "मेरा आधार", "मेरी पहचान",
)


def validate_verhoeff(number_str: str) -> bool:
    """Validates 12-digit Aadhaar number using standard Verhoeff dihedral group D_10 algorithm."""
    clean = re.sub(r"\D", "", number_str)
    if len(clean) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed([int(d) for d in clean])):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][item]]
    return c == 0


def _is_indic_line(line: str) -> bool:
    """Checks whether a line is predominantly composed of Indic/Devanagari characters (U+0900-U+0D7F)."""
    indic_count = len(re.findall(r"[\u0900-\u0D7F]", line))
    latin_count = len(re.findall(r"[A-Za-z]", line))
    if indic_count > 0 and indic_count >= latin_count:
        return True
    return any(line.strip().startswith(m) for m in ["पत्ता", "पता", "आत्मज", "सुपुत्र"])


def _extract_name_candidate(text_lines: list[str], dob_idx: int) -> str | None:
    """Extracts resident Latin name from the 1-3 lines immediately preceding the DOB anchor."""
    candidates = []
    start_lookback = max(0, dob_idx - NAME_LOOKBACK_LINES)
    for i in range(start_lookback, dob_idx):
        cand = text_lines[i]
        if any(kw in cand.lower() for kw in HEADER_KEYWORDS):
            continue
        clean_cand = re.sub(r"[^A-Za-z\s\.\-]", "", cand).strip()
        if len(clean_cand) >= MIN_NAME_LENGTH and not re.search(r"\b(DOB|MALE|FEMALE|YEAR)\b", clean_cand, re.IGNORECASE):
            candidates.append(clean_cand)
    return candidates[-1] if candidates else None


def _extract_clean_address(text_lines: list[str], pin_code: str | None) -> str | None:
    """Isolates the continuous English address block on bilingual cards, ignoring Devanagari lines."""
    english_markers = ("address:", "address", "c/o", "s/o", "w/o", "d/o")
    start_idx = -1

    # Locate starting English address line
    for idx, t in enumerate(text_lines):
        clean_t = t.lower().strip()
        if not _is_indic_line(t) and any(clean_t.startswith(m) or m in clean_t for m in english_markers):
            start_idx = idx
            break

    # Fallback to regional address marker if no English marker was present
    if start_idx == -1:
        for idx, t in enumerate(text_lines):
            clean_t = t.lower().strip()
            if any(clean_t.startswith(m) or m in clean_t for m in ("पता:", "पता", "पत्ता:", "पत्ता")):
                start_idx = idx
                break

    if start_idx < 0:
        return None

    is_eng_target = not _is_indic_line(text_lines[start_idx])
    parts = []
    for nxt in range(start_idx, min(start_idx + MAX_ADDRESS_LINES, len(text_lines))):
        nxt_line = text_lines[nxt].strip()
        # Halt at Aadhaar number (starts with 2-9)
        if re.search(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b", nxt_line):
            break
        # Halt at UIDAI contact / helpline footer
        if re.search(r"\b(help@uidai|1947|www\.uidai|authority of india)\b", nxt_line, re.IGNORECASE):
            break
        # Filter out interleaved regional language lines when parsing English address
        if is_eng_target and _is_indic_line(nxt_line):
            continue

        clean_line = re.sub(r"[\u0900-\u0D7F]+", "", nxt_line).strip() if is_eng_target else nxt_line
        clean_line = re.sub(r"\s+", " ", clean_line)
        if clean_line:
            parts.append(clean_line)
            if pin_code and pin_code in clean_line:
                break

    if not parts:
        return None

    joined = ", ".join(parts)
    joined = re.sub(r"[,;\s]*,[,;\s]*", ", ", joined)  # Normalize double commas and spaces
    return re.sub(r"^[,;\s]+|[,;\s]+$", "", joined)


class AadhaarExtractorService:
    """Extracts, normalizes, and merges demographic identity attributes from OCR lines."""

    def extract_from_lines(self, lines: list[OCRLine], full_text: str) -> AadhaarExtractedData:
        data = AadhaarExtractedData()

        # 1. Aadhaar Number: UIDAI specification dictates first digit is never 0 or 1 ([2-9])
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

        # 4. PIN Code: Indian postal PIN is a 6-digit code starting with 1-9
        if pin_matches := re.findall(r"\b([1-9][0-9]{5})\b", full_text):
            data.pin_code = pin_matches[-1]

        # 5. Name: heuristic scan 1-3 lines above the DOB / Year of Birth anchor
        text_lines = [line.text.strip() for line in lines if line.text.strip()]
        dob_idx = next((i for i, t in enumerate(text_lines) if re.search(r"(?:DOB|जन्म|Date of Birth|Year of Birth|Year of Bith|वर्ष|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})", t, re.IGNORECASE)), -1)
        if dob_idx > 0:
            data.name = _extract_name_candidate(text_lines, dob_idx)

        # 6. Address: strictly prioritize continuous English block on bilingual cards
        data.address = _extract_clean_address(text_lines, data.pin_code)

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
