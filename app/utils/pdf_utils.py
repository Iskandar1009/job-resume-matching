import os
import logging
import pypdfium2

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from PDF using multiple fallback methods.
    Tries pypdfium2 first, then PyPDF2 if that fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    if os.path.getsize(path) == 0:
        raise ValueError(f"PDF file is empty: {path}")
    

    try:
        return _extract_with_pypdfium2(path)
    except Exception as e:
        logger.warning(f"pypdfium2 failed: {e}. Trying PyPDF2...")
    


def _extract_with_pypdfium2(path: str) -> str:
    """Extract text using pypdfium2."""
    
    
    pdf = pypdfium2.PdfDocument(path)
    text_parts = []
    
    for page in pdf:
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        if text:
            text_parts.append(text)
    
    result = "\n".join(text_parts).strip()
    
    if not result:
        raise ValueError("No text extracted from PDF")
    
    return result





def is_valid_pdf(path: str) -> bool:
    """Check if a file is a valid PDF."""
    if not os.path.exists(path):
        return False
    
    if os.path.getsize(path) < 10:
        return False
    
    # Check PDF magic number
    try:
        with open(path, 'rb') as f:
            header = f.read(5)
            return header == b'%PDF-'
    except Exception:
        return False