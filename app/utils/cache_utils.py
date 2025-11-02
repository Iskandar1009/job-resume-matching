import os
import hashlib
import logging
import tempfile
from app.utils.pdf_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)
TEXT_CACHE_DIR = "cache_texts"

try:
    os.makedirs(TEXT_CACHE_DIR, exist_ok=True)
    test_file = os.path.join(TEXT_CACHE_DIR, ".test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
except Exception as e:
    logger.error(f"Cache directory issue: {e}. Using temp directory.")
    TEXT_CACHE_DIR = tempfile.gettempdir()


def _hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_cached_text(path: str) -> str:
    h = _hash_file(path)
    cache_path = os.path.join(TEXT_CACHE_DIR, f"{h}.txt")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_text = f.read()
                if cached_text.strip():
                    return cached_text
        except Exception:
            pass

    text = extract_text_from_pdf(path)
    if not text.strip():
        raise ValueError(f"No text extracted from {path}")

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

    return text


def truncate_text(text: str, max_chars=4000) -> str:
    return text[:max_chars]
