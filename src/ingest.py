# src/ingest.py

from pathlib import Path
import json

import pdfplumber # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np
import faiss # type: ignore


def extract_pages(pdf_path: Path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            pages.append({"page_num": i + 1, "text": text})
    return pages


def chunk_pages(
    pages,
    chunk_char_len=1200,
    chunk_overlap=200,
    doc_id="doc1",
    title=None,
):
    chunks = []
    buffer = ""
    start_page = None
    chunk_id = 0

    for page in pages:
        page_text = page["text"]
        page_num = page["page_num"]

        if not page_text.strip():
            continue

        if start_page is None:
            start_page = page_num

        buffer += f" [Page {page_num}] " + page_text

        while len(buffer) >= chunk_char_len:
            text_chunk = buffer[:chunk_char_len]
            chunks.append(
                {
                    "doc_id": doc_id,
                    "title": title or doc_id,
                    "chunk_id": chunk_id,
                    "page_start": start_page,
                    "page_end": page_num,
                    "text": text_chunk,
                }
            )
            chunk_id += 1
            buffer = buffer[chunk_char_len - chunk_overlap :]
            start_page = page_num

    if buffer.strip():
        chunks.append(
            {
                "doc_id": doc_id,
                "title": title or doc_id,
                "chunk_id": chunk_id,
                "page_start": start_page,
                "page_end": pages[-1]["page_num"],
                "text": buffer,
            }
        )

    return chunks


def build_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return emb, model_name


def build_faiss_index(embeddings: np.ndarray):
    emb = embeddings.astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index


def save_index(index, embeddings, chunks, base_path: Path, embed_model_name: str):
    base_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(base_path / "index.faiss"))
    np.save(base_path / "embeddings.npy", embeddings)
    meta = {
        "chunks": chunks,
        "embed_model_name": embed_model_name,
    }
    with open(base_path / "metadata.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data_dir = Path("data")

    # collect all pdfs in data
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {data_dir}")

    all_chunks = []

    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem      # "my_paper_1" or "my_paper_2"
        title = doc_id
        print(f"Processing {pdf_path.name} as doc_id={doc_id}")
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, doc_id=doc_id, title=title)
        all_chunks.extend(chunks)

    emb, model_name = build_embeddings(all_chunks)
    index = build_faiss_index(emb)
    save_index(index, emb, all_chunks, data_dir, model_name)

    print(f"Processed {len(pdf_paths)} PDFs and created {len(all_chunks)} chunks")