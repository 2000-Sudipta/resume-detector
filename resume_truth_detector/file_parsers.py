from __future__ import annotations

import io
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def extract_text_from_upload(file_storage) -> str:
    suffix = Path(file_storage.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Use one of: {supported}")

    file_bytes = file_storage.read()
    file_storage.seek(0)

    if suffix in {".txt", ".md"}:
        return file_bytes.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        from docx import Document

        document = Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return ""
