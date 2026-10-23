"""
test_document_processor.py - Test Suite for Document Processing and Metadata Extraction
"""

import os
import unittest
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_processor import (
    extract_pages_from_pdf,
    split_text_into_chunks,
    get_document_hash,
    load_and_process_pdf,
    detect_sections_in_text,
    detect_legal_provisions
)

PDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Manual_of_Designs_Practice_and_Procedure.pdf"))


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(PDF_PATH), f"Manual PDF must exist at {PDF_PATH}")

    def test_document_hash(self):
        doc_hash = get_document_hash(PDF_PATH)
        self.assertIsInstance(doc_hash, str)
        self.assertEqual(len(doc_hash), 64, "SHA-256 hash must be 64 characters long")

    def test_extract_pages_count(self):
        pages = extract_pages_from_pdf(PDF_PATH)
        self.assertEqual(len(pages), 87, "Manual must have exactly 87 pages")
        for i, page in enumerate(pages):
            self.assertEqual(page["page_number"], i + 1, "Page numbers must be 1-indexed and contiguous")

    def test_chapter_and_section_metadata(self):
        pages = extract_pages_from_pdf(PDF_PATH)
        # Page 5 should be Chapter 1 – Introduction
        self.assertIn("Chapter 1", pages[4]["chapter"])
        # Page 7 should be Chapter 2 – Important Definitions
        self.assertIn("Chapter 2", pages[6]["chapter"])
        # Page 20 should be Chapter 3 (Locarno)
        self.assertIn("Chapter 3", pages[19]["chapter"])
        # Page 36 should be Chapter 6 (Period of protection)
        self.assertIn("Chapter 6", pages[35]["chapter"])

    def test_chunking_and_metadata_preservation(self):
        chunks, summary = load_and_process_pdf(PDF_PATH, chunk_size=800, chunk_overlap=150)
        self.assertGreater(len(chunks), 150, "Should generate more than 150 chunks for 87 pages")
        self.assertEqual(summary["total_pages"], 87)

        # Verify metadata fields exist on every chunk
        for chunk in chunks:
            meta = chunk["metadata"]
            self.assertIn("page", meta)
            self.assertIn("chapter", meta)
            self.assertIn("section", meta)
            self.assertIn("chunk_id", meta)
            self.assertIn("source", meta)
            self.assertGreaterEqual(meta["page"], 1)
            self.assertLessEqual(meta["page"], 87)

    def test_empty_page_handling(self):
        dummy_pages = [
            {
                "page_number": 999,
                "chapter": "Test Chapter",
                "sections": [],
                "legal_refs": [],
                "cleaned_text": "",
                "is_empty": True
            }
        ]
        chunks = split_text_into_chunks(dummy_pages)
        self.assertEqual(len(chunks), 0, "Empty page should not generate any chunks")

    def test_regex_detectors(self):
        sample_text = "06.04 Period of protection. Under Section 11 and Rule 24, registration lasts 10 years."
        sections = detect_sections_in_text(sample_text)
        provisions = detect_legal_provisions(sample_text)

        self.assertTrue(any("Period of protection" in s for s in sections))
        self.assertTrue(any("Section 11" in p for p in provisions))
        self.assertTrue(any("Rule 24" in p for p in provisions))

    def test_rag_retrieval_and_citations(self):
        from rag_pipeline import DesignsLawRAG
        rag = DesignsLawRAG()
        # Ensure document is indexed
        status = rag.get_index_status()
        if not status["is_indexed"]:
            rag.index_document(PDF_PATH)

        # Retrieve for Q1 (Copyright duration)
        chunks = rag.retrieve("How long does a proprietor have copyright in a registered design, and can it be extended?", k=3)
        self.assertGreater(len(chunks), 0, "Must retrieve relevant chunks for copyright duration")
        pages = [c["page"] for c in chunks]
        self.assertTrue(36 in pages or 39 in pages, "Retrieved pages must include page 36 or 39 for copyright duration")

    def test_negative_question_not_found(self):
        from rag_pipeline import DesignsLawRAG
        rag = DesignsLawRAG()
        # Q3 Trademark fee must return exactly "Not found"
        ans = rag.answer_question("What is the government fee to register a trademark?")
        self.assertEqual(ans["answer"], "Not found")
        self.assertTrue(ans["is_not_found"])

        # Q4 Current Controller General must return "Not found"
        ans_cg = rag.answer_question("Who is the current Controller General of Patents?")
        self.assertEqual(ans_cg["answer"], "Not found")
        self.assertTrue(ans_cg["is_not_found"])

    def test_missing_api_key_handling(self):
        from rag_pipeline import DesignsLawRAG
        rag_no_key = DesignsLawRAG(api_key="")
        rag_no_key.api_key = ""
        # Should gracefully return an error dict rather than crashing
        res = rag_no_key._call_gemini("Test prompt")
        self.assertTrue(res.startswith("ERROR: Google Gemini API key is missing"))


if __name__ == "__main__":
    unittest.main()
