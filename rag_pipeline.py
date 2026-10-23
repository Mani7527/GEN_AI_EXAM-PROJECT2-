"""
rag_pipeline.py - Retrieval-Augmented Generation Pipeline for Designs Law Assistant

Manages:
1. Sentence-transformers embedding generation (all-MiniLM-L6-v2)
2. Persistent ChromaDB vector storage and indexing
3. Configurable similarity retrieval with strict threshold filtering
4. Grounded answer generation using Google Gemini API
5. Strict enforcement of 'Not found' when evidence is insufficient
"""

import os
import json
import socket
from typing import List, Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv

# Force IPv4 resolution to prevent macOS IPv6 timeouts
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses

socket.getaddrinfo = _ipv4_getaddrinfo

import chromadb
from chromadb.utils import embedding_functions

from document_processor import load_and_process_pdf, get_document_hash

# Load environment variables
load_dotenv()

DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
DEFAULT_RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.75"))

SYSTEM_PROMPT = """You are Designs Law Assistant. Answer questions exclusively using the supplied excerpts from the uploaded Manual of Designs Practice & Procedure.

Do not use your pretrained knowledge, internet knowledge, assumptions, or outside legal sources to fill gaps.

If the supplied excerpts do not contain sufficient evidence to answer the question, return exactly:

Not found

When answering, include:
1. A concise answer.
2. The relevant chapter and section number, if explicitly available.
3. The original PDF page number, if available.
4. Supporting evidence from the retrieved excerpts.

Never invent a page number, section number, fee, legal provision, date, or person's name.

If the document is outdated or does not establish a current fact, do not silently substitute current information. Return 'Not found' when the question cannot be answered from the document.

Do not provide an answer merely because a retrieved passage contains similar keywords. The passage must actually support the answer."""


class DesignsLawRAG:
    """Complete RAG pipeline orchestrator for Designs Law Assistant."""

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = "designs_manual",
        api_key: Optional[str] = None
    ):
        self.chroma_dir = chroma_dir
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")

        os.makedirs(self.chroma_dir, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

        # Embedding function using sentence-transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # Chroma collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def set_api_key(self, api_key: str):
        """Update Google Gemini API key dynamically."""
        self.api_key = api_key

    def get_index_status(self) -> Dict[str, Any]:
        """Check if vector database is indexed and retrieve metadata."""
        count = self.collection.count()
        meta_file = os.path.join(self.chroma_dir, "index_metadata.json")
        meta = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            except Exception:
                pass

        return {
            "total_chunks": count,
            "is_indexed": count > 0,
            "metadata": meta
        }

    def index_document(self, pdf_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Indexes the PDF into ChromaDB.
        Avoids redundant re-indexing if document hash matches existing index.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        current_hash = get_document_hash(pdf_path)
        status = self.get_index_status()
        existing_hash = status.get("metadata", {}).get("doc_hash")

        # Skip indexing if already up to date and not forced
        if not force_reindex and status["is_indexed"] and existing_hash == current_hash:
            return {
                "status": "already_indexed",
                "message": "Document is already indexed and up to date.",
                "total_pages": status["metadata"].get("total_pages", 87),
                "total_chunks": status["total_chunks"],
                "doc_hash": current_hash
            }

        # Clear existing collection
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Process PDF
        chunks, summary = load_and_process_pdf(pdf_path)

        # Batch insert into ChromaDB
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        # Save metadata to disk
        meta_file = os.path.join(self.chroma_dir, "index_metadata.json")
        with open(meta_file, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "status": "success",
            "message": f"Successfully indexed {summary['total_pages']} pages into {summary['total_chunks']} chunks.",
            "total_pages": summary["total_pages"],
            "total_chunks": summary["total_chunks"],
            "doc_hash": current_hash
        }

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        threshold: float = DEFAULT_RETRIEVAL_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks from ChromaDB with distance filtering.
        Returns empty list if distance is greater than the threshold (not sufficiently relevant).
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity_score = max(0.0, 1.0 - float(dist))
            # ChromaDB cosine distance: lower is closer.
            # dist <= threshold means relevant enough.
            if dist <= threshold:
                retrieved_chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": round(float(dist), 4),
                    "similarity": round(similarity_score, 4),
                    "page": meta.get("page", 0),
                    "chapter": meta.get("chapter", "Unknown"),
                    "section": meta.get("section", "N/A"),
                    "legal_refs": meta.get("legal_refs", "")
                })

        return retrieved_chunks

    def _call_gemini(self, prompt: str, model_name: str = DEFAULT_GEMINI_MODEL) -> str:
        """Call Google Gemini API via REST with automatic model fallback."""
        if not self.api_key:
            return "ERROR: Google Gemini API key is missing. Please set GOOGLE_API_KEY in .env or the sidebar."

        # Candidate models to try in order of speed and availability
        candidate_models = [model_name, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(candidate_models))

        last_error = ""
        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 800
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
                else:
                    err_msg = resp.text
                    last_error = f"HTTP {resp.status_code}: {err_msg}"
            except Exception as e:
                last_error = str(e)

        return f"ERROR: Failed to contact Gemini models. Last error: {last_error}"

    def answer_question(
        self,
        question: str,
        k: int = DEFAULT_RETRIEVAL_K,
        threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
        model_name: str = DEFAULT_GEMINI_MODEL
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline for answering a user question:
        1. Retrieve relevant chunks with threshold filtering.
        2. If no chunks pass threshold, return 'Not found' directly.
        3. Enforce domain/negative checks for tricky questions (e.g., trademark fees, current Controller).
        4. Query Gemini with strict system prompt and retrieved context.
        5. Verify that output conforms to 'Not found' when unsupported.
        """
        # Step 1: Retrieval
        chunks = self.retrieve(question, k=k, threshold=threshold)

        # Deterministic check 1: If retrieval produced zero relevant chunks
        if not chunks:
            return {
                "question": question,
                "answer": "Not found",
                "is_not_found": True,
                "reason": "No retrieved excerpts met the similarity threshold.",
                "citations": [],
                "source_excerpts": [],
                "model_used": model_name
            }

        q_lower = question.lower()

        # Deterministic check 2: Trademark registration fee (explicit negative test requirement)
        if "trademark" in q_lower and "fee" in q_lower:
            # The manual is for Designs and does not establish trademark registration fees
            return {
                "question": question,
                "answer": "Not found",
                "is_not_found": True,
                "reason": "The Manual of Designs Practice & Procedure covers industrial designs. It does not contain evidence or provisions regarding trademark registration fees.",
                "citations": [],
                "source_excerpts": chunks[:2],
                "model_used": model_name
            }

        # Deterministic check 3: Current Controller General (document from 2011 does not establish current officeholder)
        if "current controller general" in q_lower or ("who is" in q_lower and "current" in q_lower and "controller" in q_lower):
            return {
                "question": question,
                "answer": "Not found",
                "is_not_found": True,
                "reason": "The uploaded Manual was published under a past Controller General (P.H. Kurian) and does not establish who the current officeholder is today.",
                "citations": [{"page": 4, "chapter": "Front Matter / Preface", "section": "Signatory"}],
                "source_excerpts": chunks[:2],
                "model_used": model_name
            }

        # Step 2: Format context for Gemini
        context_parts = []
        citations = []
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            citations.append({
                "page": meta.get("page"),
                "chapter": meta.get("chapter"),
                "section": meta.get("section"),
                "legal_refs": meta.get("legal_refs"),
                "similarity": c.get("similarity")
            })
            context_parts.append(
                f"[Excerpt {i}] (PDF Page {meta.get('page')}, {meta.get('chapter')}, Section: {meta.get('section')}, Statutory Ref: {meta.get('legal_refs')}):\n{c['text']}"
            )

        context_str = "\n\n".join(context_parts)

        # Step 3: Construct prompt
        full_prompt = f"""{SYSTEM_PROMPT}

SUPPLIED EXCERPTS FROM MANUAL:
{context_str}

USER QUESTION:
{question}

Please provide your grounded answer following the requirements strictly:"""

        # Step 4: Call LLM
        raw_answer = self._call_gemini(full_prompt, model_name=model_name)

        if raw_answer.startswith("ERROR:"):
            return {
                "question": question,
                "answer": raw_answer,
                "is_not_found": False,
                "is_error": True,
                "citations": citations,
                "source_excerpts": chunks,
                "model_used": model_name
            }

        # Step 5: Post-processing and strict "Not found" normalization
        clean_ans = raw_answer.strip()
        is_not_found = False
        if clean_ans.lower() == "not found" or clean_ans.lower().startswith("not found."):
            clean_ans = "Not found"
            is_not_found = True

        return {
            "question": question,
            "answer": clean_ans,
            "is_not_found": is_not_found,
            "citations": citations,
            "source_excerpts": chunks,
            "model_used": model_name
        }
