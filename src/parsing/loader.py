"""
Document loader module for SmartHire GenAI.
Supports extracting clean text from PDF and DOCX resume files,
with support for both local file paths and in-memory file buffers (e.g., Streamlit uploads).
"""

import io
import re
from pathlib import Path
from typing import Union, List, Optional
import pypdf
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    # Replace multiple spaces/newlines with single equivalents
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(file_source: Union[str, Path, io.BytesIO]) -> str:
    """
    Extract text content from a PDF file path or byte stream.
    """
    extracted_text = []
    try:
        reader = pypdf.PdfReader(file_source)
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)
    except Exception as e:
        raise ValueError(f"Failed to read PDF document: {str(e)}")

    return clean_text("\n\n".join(extracted_text))


def load_docx(file_source: Union[str, Path, io.BytesIO]) -> str:
    """
    Extract text content from a DOCX file path or byte stream.
    Includes both paragraphs and table cell contents.
    """
    extracted_text = []
    try:
        doc = docx.Document(file_source)
        for para in doc.paragraphs:
            if para.text.strip():
                extracted_text.append(para.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    extracted_text.append(" | ".join(row_cells))
    except Exception as e:
        raise ValueError(f"Failed to read DOCX document: {str(e)}")

    return clean_text("\n".join(extracted_text))


def load_document(file_source: Union[str, Path, io.BytesIO], filename: Optional[str] = None) -> str:
    """
    Universal loader for PDF and DOCX documents.
    Accepts a filepath string/Path, or a file-like object with an optional filename.
    """
    name = ""
    if isinstance(file_source, (str, Path)):
        name = str(file_source).lower()
    elif filename:
        name = filename.lower()
    elif hasattr(file_source, "name"):
        name = file_source.name.lower()

    if name.endswith(".pdf"):
        return load_pdf(file_source)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return load_docx(file_source)
    elif name.endswith(".txt") or name.endswith(".md"):
        if isinstance(file_source, (str, Path)):
            with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read())
        else:
            return clean_text(file_source.read().decode("utf-8", errors="ignore"))
    else:
        # Default attempt PDF then DOCX
        try:
            return load_pdf(file_source)
        except Exception:
            return load_docx(file_source)


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into chunks using recursive character splitting.
    Useful for breaking large documents into retrievable fragments.
    """
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)
