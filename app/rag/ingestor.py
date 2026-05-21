import json
import re
from pathlib import Path
from typing import Iterator
import logging

logger = logging.getLogger(__name__)

def load_mimic_jsonl(path: str) -> Iterator[dict]:
    """
    Load MIMIC-IV discharge summaries from a .jsonl file.
    Each line : {"note_id": "...", "text": "...", "icd10_codes": [...]}
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIMIC data not found at: {path}")
    
    with open (p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def chunk_records(record: dict, max_chars: int = 1200) -> list[dict]:
    """
    Split a long clinical note into overlapping chunks for better retrieval.
    Each chunk becomes one vector store entry.

    Returns list of chunk dicts ready for embedding + insertion.
    """

    text = record.get("text", "").strip()
    text = _clean_clinical_text(text)
    note_id = record.get("note_id", "unknown")
    icd10_codes = record.get("icd10_codes", [])

    if len(text) < max_chars:
        return [{
            "content": text,
            "source": "mimim_iv",
            "icd10_codes": icd10_codes,
            "metadata": {
                "note_id": note_id,
                "chunk_index": 0,
                "total_chunks": 1,
            }
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    overlap = 200


    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end]

        #Try to break on sentence boundary
        if end < len(text):
            last_period = chunk_text.rfind(". ")
            if last_period > max_chars // 2:
                chunk_text = chunk_text[:last_period + 1]
                end = start + last_period + 1
            
            chunks.append({
                "content": chunk_text.strip(),
                "source": "mimic_iv",
                "icd10_codes": icd10_codes,
                "metadata": {
                    "note_id": note_id,
                    "chunk_index": chunk_index
                }
            })

            start = end - overlap
            chunk_index += 1
    
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks

def _clean_clinical_text(text: str) -> str:
    """Remove common MIMIC de-identification artifacts and noise."""
    # Remove [** ... **] de-identification markers
    text = re.sub(r'\[\*\*.*?\*\*\]', '[REDACTED]', text)
    # Collapse excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
    


