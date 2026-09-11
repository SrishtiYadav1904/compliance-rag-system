import re
from typing import Dict, Any, Tuple


class LanguageDetector:
    """
    Fast, deterministic language detector for Legal Metrology queries.
    Accurately classifies:
    - English ('en')
    - Hindi ('hi')
    - Marathi ('mr')
    - Tamil ('ta')
    """

    TAMIL_REGEX = re.compile(r"[\u0B80-\u0BFF]")
    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
    ENGLISH_REGEX = re.compile(r"[a-zA-Z]")

    # Characteristic Marathi words and morphemes
    MARATHI_MARKERS = {
        "आहे", "आहेत", "नाही", "करावे", "करावी", "पॅकेजवर", "कशी", "कोणत्या",
        "तपशील", "किंमत", "वस्तूवर", "बदलण्यास", "परवानगी", "तसेच", "च्या",
        "तील", "मधील", "साठी", "केले", "होते", "किरकोळ", "कमाल", "घोषणा", "पाहिजे"
    }

    # Characteristic Hindi words and morphemes
    HINDI_MARKERS = {
        "है", "हैं", "होना", "चाहिए", "अनिवार्य", "कौन", "कौन-कौन", "सी", "सा",
        "का", "के", "की", "में", "पर", "लिए", "बदलने", "सकते", "सकता", "सकती",
        "किया", "गया", "जाएगा", "खुदरा", "मूल्य", "अधिकतम", "स्टीकर", "मात्रा"
    }

    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detects language of input text.
        Returns: (lang_code, confidence)
        """
        if not text or not text.strip():
            return "en", 1.0

        # Check Tamil script
        tamil_matches = len(self.TAMIL_REGEX.findall(text))
        if tamil_matches > 2:
            return "ta", 0.99

        # Check Devanagari script (Hindi or Marathi)
        devanagari_matches = len(self.DEVANAGARI_REGEX.findall(text))
        if devanagari_matches > 2:
            # Check for Marathi specific retroflex 'ळ' (U+0933)
            if "\u0933" in text:
                return "mr", 0.98

            words = set(re.findall(r"[\u0900-\u097F]+", text))
            marathi_score = sum(1 for w in words if w in self.MARATHI_MARKERS)
            hindi_score = sum(1 for w in words if w in self.HINDI_MARKERS)

            if marathi_score > hindi_score:
                return "mr", 0.95
            elif hindi_score > marathi_score:
                return "hi", 0.95
            else:
                # Default Devanagari to Hindi if markers equal or ambiguous
                return "hi", 0.80

        # Check Latin / English
        latin_matches = len(self.ENGLISH_REGEX.findall(text))
        if latin_matches > 0:
            return "en", 0.95

        return "en", 0.50

    def get_language_name(self, code: str) -> str:
        names = {
            "en": "English",
            "hi": "Hindi",
            "mr": "Marathi",
            "ta": "Tamil"
        }
        return names.get(code, "English")
