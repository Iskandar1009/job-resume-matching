# pdf_utils.py
import pypdfium2

def extract_text_from_pdf(path: str) -> str:
    pdf = pypdfium2.PdfDocument(path)
    text = []
    for page in pdf:
        text.append(page.get_textpage().get_text_bounded())
    return "\n".join(text).strip()
