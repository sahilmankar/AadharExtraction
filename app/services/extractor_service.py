import re
from typing import Literal
from app.schemas.document import AadhaarExtractedData, OCRLine

# Verhoeff algorithm tables for Aadhaar 12-digit checksum validation
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


def validate_verhoeff(number_str: str) -> bool:
    """Validates 12-digit Aadhaar number using the standard Verhoeff checksum algorithm."""
    clean = re.sub(r"\D", "", number_str)
    if len(clean) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed([int(d) for d in clean])):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][item]]
    return c == 0


class AadhaarExtractorService:
    def extract_from_lines(self, lines: list[OCRLine], full_text: str) -> AadhaarExtractedData:
        data = AadhaarExtractedData()

        # 1. Aadhaar Number Extraction (regular or masked)
        masked_match = re.search(r"\b([X\*\.]{4}[\s\-]?[X\*\.]{4}[\s\-]?[0-9]{4})\b", full_text, re.IGNORECASE)
        unmasked_match = re.search(r"\b([2-9][0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4})\b", full_text)

        if masked_match:
            raw = masked_match.group(1).upper()
            clean = re.sub(r"[\s\-]", "", raw)
            formatted = f"{clean[:4]} {clean[4:8]} {clean[8:12]}"
            data.aadhaar_number = formatted
            data.is_masked = True
            data.is_aadhaar_number_valid = True
        elif unmasked_match:
            raw = unmasked_match.group(1)
            clean = re.sub(r"[\s\-]", "", raw)
            formatted = f"{clean[:4]} {clean[4:8]} {clean[8:12]}"
            data.aadhaar_number = formatted
            data.is_masked = False
            data.is_aadhaar_number_valid = validate_verhoeff(clean)

        # 2. Date of Birth or Year of Birth
        dob_match = re.search(
            r"(?:DOB|Date of Birth|जन्म\s*तारीख|जन्म\s*तिथि)[\s\:\-]+([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})",
            full_text,
            re.IGNORECASE,
        )
        if not dob_match:
            dob_match = re.search(r"\b([0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b", full_text)

        if dob_match:
            data.dob = dob_match.group(1)
        else:
            yob_match = re.search(r"(?:Year of Birth|जन्म\s*वर्ष)[\s\:\-]+([12][0-9]{3})", full_text, re.IGNORECASE)
            if yob_match:
                data.dob = yob_match.group(1)

        # 3. Gender
        gender_match = re.search(r"\b(MALE|FEMALE|TRANSGENDER|पुरुष|महिला)\b", full_text, re.IGNORECASE)
        if gender_match:
            g = gender_match.group(1).upper()
            if g in ["पुरुष", "MALE"]:
                data.gender = "MALE"
            elif g in ["महिला", "FEMALE"]:
                data.gender = "FEMALE"
            elif "TRANS" in g:
                data.gender = "TRANSGENDER"

        # 4. PIN Code
        pin_matches = re.findall(r"\b([1-9][0-9]{5})\b", full_text)
        if pin_matches:
            data.pin_code = pin_matches[-1]

        # 5. Name heuristics
        # Look in lines before DOB
        text_lines = [line.text.strip() for line in lines if line.text.strip()]
        dob_index = -1
        for idx, t in enumerate(text_lines):
            if re.search(r"(?:DOB|जन्म|Date of Birth|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})", t, re.IGNORECASE):
                dob_index = idx
                break

        header_keywords = [
            "government", "india", "authority", "unique", "identification",
            "aadhaar", "आधार", "भारत", "सरकार", "मेरा आधार", "मेरी पहचान"
        ]

        if dob_index > 0:
            candidate_names = []
            for i in range(max(0, dob_index - 3), dob_index):
                candidate = text_lines[i]
                cand_lower = candidate.lower()
                if any(kw in cand_lower for kw in header_keywords):
                    continue
                # Clean candidate
                clean_cand = re.sub(r"[^A-Za-z\s\.\-]", "", candidate).strip()
                if len(clean_cand) >= 3 and not re.search(r"\b(DOB|MALE|FEMALE|YEAR)\b", clean_cand, re.IGNORECASE):
                    candidate_names.append(clean_cand)
            if candidate_names:
                data.name = candidate_names[-1]

        # 6. Address heuristics
        address_markers = ["address:", "पता:", "address", "पता"]
        for idx, t in enumerate(text_lines):
            t_lower = t.lower()
            if any(t_lower.startswith(marker) for marker in address_markers) or "c/o" in t_lower or "s/o" in t_lower or "w/o" in t_lower or "d/o" in t_lower:
                addr_parts = [t]
                for nxt in range(idx + 1, min(idx + 5, len(text_lines))):
                    nxt_line = text_lines[nxt]
                    if re.search(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b", nxt_line):
                        break
                    addr_parts.append(nxt_line)
                    if data.pin_code and data.pin_code in nxt_line:
                        break
                data.address = ", ".join(addr_parts)
                break

        # 7. Document side detection
        has_front_indicators = bool(data.dob or data.gender or re.search(r"\b(dob|male|female|जन्म)\b", full_text, re.IGNORECASE))
        has_back_indicators = bool(data.address or re.search(r"\b(address|पता|help@uidai|1947)\b", full_text, re.IGNORECASE))

        if has_front_indicators and has_back_indicators:
            data.document_side = "FULL_PAGE"
        elif has_front_indicators:
            data.document_side = "FRONT"
        elif has_back_indicators:
            data.document_side = "BACK"
        else:
            data.document_side = "UNKNOWN"

        return data

    def merge_extracted_data(self, data_list: list[AadhaarExtractedData]) -> AadhaarExtractedData:
        """Merges extracted data from multiple pages or front/back images."""
        merged = AadhaarExtractedData()
        for d in data_list:
            if not merged.aadhaar_number and d.aadhaar_number:
                merged.aadhaar_number = d.aadhaar_number
                merged.is_aadhaar_number_valid = d.is_aadhaar_number_valid
                merged.is_masked = d.is_masked
            elif merged.aadhaar_number and merged.is_masked and not d.is_masked and d.aadhaar_number:
                merged.aadhaar_number = d.aadhaar_number
                merged.is_aadhaar_number_valid = d.is_aadhaar_number_valid
                merged.is_masked = False

            if not merged.name and d.name:
                merged.name = d.name
            if not merged.dob and d.dob:
                merged.dob = d.dob
            if not merged.gender and d.gender:
                merged.gender = d.gender
            if not merged.address and d.address:
                merged.address = d.address
            if not merged.pin_code and d.pin_code:
                merged.pin_code = d.pin_code

        sides = {d.document_side for d in data_list}
        if "FULL_PAGE" in sides or ("FRONT" in sides and "BACK" in sides):
            merged.document_side = "FULL_PAGE"
        elif "FRONT" in sides:
            merged.document_side = "FRONT"
        elif "BACK" in sides:
            merged.document_side = "BACK"
        else:
            merged.document_side = "UNKNOWN"

        return merged

    def classify_document(self, merged: AadhaarExtractedData, full_text: str) -> Literal["AADHAAR_FRONT", "AADHAAR_BACK", "E_AADHAAR", "NOT_CLASSIFIED"]:
        has_aadhaar_keyword = bool(
            re.search(r"(aadhaar|आधार|uidai|unique identification authority|भारत सरकार|government of india)", full_text, re.IGNORECASE)
            or merged.aadhaar_number
        )
        if not has_aadhaar_keyword:
            return "NOT_CLASSIFIED"

        if merged.document_side == "FULL_PAGE" or (merged.dob and merged.address):
            return "E_AADHAAR"
        elif merged.document_side == "FRONT" or (merged.dob and not merged.address):
            return "AADHAAR_FRONT"
        elif merged.document_side == "BACK" or (merged.address and not merged.dob):
            return "AADHAAR_BACK"

        return "NOT_CLASSIFIED"
