"""
document_processor.py - PDF Extraction and Chunking Pipeline

Extracts text from the 'Manual of Designs Practice & Procedure' page by page,
extracts structural metadata (chapters, sections, statutory references, page numbers),
and splits text into overlapping chunks suitable for vector retrieval in ChromaDB.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Tuple
import pypdf
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Known chapter mapping for high accuracy in Manual of Designs Practice & Procedure
KNOWN_CHAPTERS = {
    1: "Front Matter / Title Page",
    2: "Contents",
    3: "Preface",
    4: "Preface / Disclaimer",
    5: "Chapter 1 – Introduction",
    6: "Chapter 1 – Introduction (Continued)",
    7: "Chapter 2 – Important Definitions",
    8: "Chapter 2 – Important Definitions",
    9: "Chapter 2 – Important Definitions",
    10: "Chapter 2 – Important Definitions",
    11: "Chapter 3 – Filing of Design Application",
    12: "Chapter 3 – Filing of Design Application",
    13: "Chapter 3 – Filing of Design Application",
    14: "Chapter 3 – Contents of Application & Classification",
    15: "Chapter 3 – Representation & Power of Attorney",
    16: "Chapter 3 – Priority Document & Representation Sheet",
    17: "Chapter 3 – Representation Sheet Requirements",
    18: "Chapter 3 – Representation Sheet Requirements",
    19: "Chapter 3 – Representation Sheet (Drawings/Colours)",
    20: "Chapter 3 – Classification of Designs (Locarno)",
    21: "Chapter 3 – Address for Service & Serving Documents",
    22: "Chapter 3 – Fees & Document Submission",
    23: "Chapter 3 – Initial Processing of Application",
    24: "Chapter 4 – Examination and Registration of Designs",
    25: "Chapter 4 – Substantive Examination",
    26: "Chapter 4 – What Constitutes a Design",
    27: "Chapter 4 – Novelty and Originality",
    28: "Chapter 4 – Novelty Search & Originality",
    29: "Chapter 4 – Prior Registered Designs",
    30: "Chapter 4 – Public Order, Morality & Security of India",
    31: "Chapter 4 – Consideration of Report of Examiner",
    32: "Chapter 4 – Objections, Hearings & Registration",
    33: "Chapter 4 – Register of Designs & Certificate of Registration",
    34: "Chapter 5 – The Patent Office Journal",
    35: "Chapter 6 – Piracy of Registered Designs and Consequences",
    36: "Chapter 6 – Period of Protection (Copyright Duration)",
    37: "Chapter 7 – Post Registration Procedures (Cancellation)",
    38: "Chapter 7 – Procedure for Cancellation",
    39: "Chapter 7 – Extension of Copyright (Renewal)",
    40: "Chapter 7 – Restoration of Lapsed Design",
    41: "Chapter 7 – Restoration Procedure & Unpaid Extension Fee",
    42: "Chapter 7 – Rectification of Register",
    43: "Chapter 7 – Procedure for Rectification",
    44: "Chapter 7 – Rectification Hearings & Decision",
    45: "Chapter 8 – Transfer of Rights (Assignment/Licensing)",
    46: "Chapter 8 – Registration of Title & Admissibility",
    47: "Chapter 9 – General Services (Existence of Copyright & Copies)",
    48: "Chapter 9 – Certified Copies & Inspection of Register",
    49: "Chapter 9 – Alteration of Address & Duplicate Certificate",
    50: "Chapter 10 – Use & Acquisition of Registered Design by Government",
    51: "Chapter 11 – General Powers of Controller",
    52: "Chapter 11 – Clerical Errors & General Power of Amendment",
    53: "Chapter 11 – General Power for Extension of Time",
    54: "Chapter 12 – Evidences Etc. (Affidavits)",
    55: "Chapter 12 – Form of Affidavits",
    56: "Chapter 12 – Documents & Annual Report by Controller",
    57: "Chapter 12 – Annual Report Contents",
    58: "Chapter 13 – Restrictive Conditions in Contracts Etc.",
    59: "Chapter 13 – Restrictive Conditions Void",
    60: "Chapter 14 – Appeals to High Court & Jurisdiction",
    61: "Chapter 15 – Requirements Before Delivery on Sales (Marking)",
    62: "Chapter 15 – Dispensing Marking & Exact Representation",
    63: "Chapter 16 – Timelines Prescribed Under Designs Act & Rules",
    64: "Chapter 16 – Timelines (Continued)",
    65: "Annexure I – Important Case Laws (Article)",
    66: "Annexure I – Important Case Laws (Article & Popeye)",
    67: "Annexure I – Important Case Laws (Functional Shape & Escorts)",
    68: "Annexure I – Important Case Laws (Set of Articles)",
    69: "Annexure I – Important Case Laws (Copyright Act vs Designs Act)",
    70: "Annexure I – Important Case Laws (Microfibers Inc.)",
    71: "Annexure I – Important Case Laws (Mere Mechanical Device)",
    72: "Annexure I – Important Case Laws (Judged Solely by Eye)",
    73: "Annexure I – Important Case Laws (Customer Preference & Eye Appeal)",
    74: "Annexure I – Important Case Laws (Noticeability & Trademark Exclusion)",
    75: "Annexure I – Important Case Laws (Property Mark & Pugh v. Riley)",
    76: "Annexure I – Important Case Laws (High Court & Proprietor of Design)",
    77: "Annexure I – Important Case Laws (New or Original - Brighto Auto)",
    78: "Annexure I – Important Case Laws (Gammeter & Glaxo Smithkline)",
    79: "Annexure I – Important Case Laws (Supreme Court Civil Appeal 3185/2008)",
    80: "Annexure I – Important Case Laws (Publication & Rotela Auto / Gopal Glass)",
    81: "Annexure I – Important Case Laws (Venus Industries & Good Faith Disclosure)",
    82: "Annexure I – Important Case Laws (Significantly Distinguishable - Phillips)",
    83: "Annexure I – Important Case Laws (Repeal & Saving Provision - Faber Castell)",
    84: "Annexure I – Important Case Laws (IAG Company Ltd Vs Triveni Glass)",
    85: "Annexure I – Important Case Laws (Section 48(4) Pending Proceedings)",
    86: "Annexure II – Specimen Novelty Statements & Disclaimers",
    87: "Annexure II – Specimen Representation Sheet (Tea Pot)"
}


def get_document_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file to track changes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_sections_in_text(text: str) -> List[str]:
    """Extract section titles or numbers from text (e.g., 06.04 Period of protection)."""
    # Matches patterns like '06.04 Period of protection' or '03.06.02.06 Classification'
    pattern = r'(?:\b|\n)(\d{2}\.\d{2}(?:\.\d{2})*(?:\.\d{2})?\s+[A-Za-z0-9\s,\-\(\)\&]+)'
    matches = re.findall(pattern, text)
    cleaned = []
    for m in matches:
        title = m.strip().split("\n")[0].strip()
        if len(title) > 5 and title not in cleaned:
            cleaned.append(title)
    return cleaned


def detect_legal_provisions(text: str) -> List[str]:
    """Extract references to Sections or Rules in text."""
    sec_pattern = r'\b(?:Section|Sec\.?)\s+\d+(?:\([0-9a-zA-Z]+\))*'
    rule_pattern = r'\b(?:Rule)\s+\d+(?:\([0-9a-zA-Z]+\))*'
    sections = re.findall(sec_pattern, text, re.IGNORECASE)
    rules = re.findall(rule_pattern, text, re.IGNORECASE)
    all_refs = list(dict.fromkeys(sections + rules))
    return all_refs[:8]  # Limit to top references


def extract_pages_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page by page from the PDF, preserving page numbers,
    chapters, sections, and statutory references.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_data = []

    current_chapter = "General"

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed PDF page number
        raw_text = page.extract_text() or ""
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

        # Update chapter if known or dynamically detected
        if page_num in KNOWN_CHAPTERS:
            current_chapter = KNOWN_CHAPTERS[page_num]
        else:
            ch_match = re.search(r'CHAPTER\s*[-–—]?\s*(\d+)', raw_text, re.IGNORECASE)
            if ch_match:
                current_chapter = f"Chapter {ch_match.group(1)}"

        sections_found = detect_sections_in_text(raw_text)
        legal_refs = detect_legal_provisions(raw_text)

        # Handle pages with little or no extractable text
        is_empty = len(cleaned_text) < 15

        pages_data.append({
            "page_number": page_num,
            "chapter": current_chapter,
            "sections": sections_found,
            "primary_section": sections_found[0] if sections_found else "",
            "legal_refs": legal_refs,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "is_empty": is_empty,
            "char_count": len(cleaned_text)
        })

    return pages_data


def split_text_into_chunks(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    source_filename: str = "Manual_of_Designs_Practice_and_Procedure.pdf"
) -> List[Dict[str, Any]]:
    """
    Splits page text into overlapping chunks, maintaining strict metadata:
    page number, chapter, section, and legal references.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    all_chunks = []
    chunk_counter = 0

    for page in pages_data:
        if page["is_empty"] or not page["cleaned_text"]:
            continue

        page_num = page["page_number"]
        chapter = page["chapter"]
        sections = page["sections"]
        legal_refs = ", ".join(page["legal_refs"]) if page["legal_refs"] else "None explicitly cited"

        # Split the text of this specific page
        text_chunks = splitter.split_text(page["cleaned_text"])

        for sub_idx, chunk_text in enumerate(text_chunks):
            chunk_counter += 1
            chunk_id = f"page_{page_num}_chunk_{sub_idx + 1}"

            # Determine closest section for this chunk if possible
            matching_section = ""
            for sec in sections:
                if sec[:15].lower() in chunk_text.lower():
                    matching_section = sec
                    break
            if not matching_section and sections:
                matching_section = sections[0]

            metadata = {
                "source": source_filename,
                "page": page_num,
                "chapter": chapter,
                "section": matching_section or "N/A",
                "legal_refs": legal_refs,
                "chunk_id": chunk_id,
                "chunk_index": chunk_counter
            }

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })

    return all_chunks


def load_and_process_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Full document processing pipeline:
    Extracts pages -> detects metadata -> splits into chunks -> summarizes metrics.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    doc_hash = get_document_hash(pdf_path)
    file_size_bytes = os.path.getsize(pdf_path)
    filename = os.path.basename(pdf_path)

    pages = extract_pages_from_pdf(pdf_path)
    chunks = split_text_into_chunks(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_filename=filename
    )

    summary = {
        "filename": filename,
        "file_path": pdf_path,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "doc_hash": doc_hash,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "empty_pages_count": sum(1 for p in pages if p["is_empty"])
    }

    return chunks, summary
