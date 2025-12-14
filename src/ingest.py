# src/ingest.py

from pathlib import Path
import json

import pdfplumber # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np
import faiss # type: ignore


# opens one PDF with pdfplumber, extracts text page by page, 
# cleans whitespace, 
# and returns a list of {page_num, text} records.
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
    
    # Start with an empty buffer
    chunks = []
    buffer = ""
    start_page = None
    chunk_id = 0

    # For each page, append the page’s text into the buffer
    for page in pages:
        page_text = page["text"]
        page_num = page["page_num"]

        if not page_text.strip():
            continue

        if start_page is None:
            start_page = page_num

        buffer += f" [Page {page_num}] " + page_text

        # A single page can easily exceed 1200 characters.
        # keep cutting chunks off the front of buffer 
        # until buffer becomes shorter than one chunk
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

    #If buffer contains at least one non-whitespace character, append one last chunk.
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


# turns each text chunk into a numeric vector so FAISS can do similarity search.
def build_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # loads an embedding model from sentence-transformers.
    model = SentenceTransformer(model_name)
    # extracts only the raw text from each chunk.
    texts = [c["text"] for c in chunks]
    # runs the embedding model on every text element in texts
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
   # returns embedding matrix and model_name
    return emb, model_name


def build_faiss_index(embeddings: np.ndarray): # expects embeddings to be a NumPy array.
    # get the embeddings and store them in 32-bit floats. FAISS commonly expects float32 vectors
    emb = embeddings.astype("float32")
    # get the length of the embedding vectors
    dim = emb.shape[1]
    # faiss.IndexFlatL2(dim) returns a FAISS index object 
    # that stores vectors of length dim.
    # FAISS index object comes with built in search function
    # we will later call search on index to get the top closest stored vectors to 
    # query vector. 
    index = faiss.IndexFlatL2(dim)
    #adds the embedding matrix to index object. 
    index.add(emb)
    return index


def save_index(index, embeddings, chunks, base_path: Path, embed_model_name: str):
    base_path.mkdir(parents=True, exist_ok=True)
    # Save the FAISS index to disk as a file named index.faiss.
    faiss.write_index(index, str(base_path / "index.faiss"))
    # Save the embeddings array to disk as embeddings.npy in NumPy’s binary format.
    np.save(base_path / "embeddings.npy", embeddings)
    # Build a meta dict that we will write to JSON
    meta = {
        "chunks": chunks,
        "embed_model_name": embed_model_name,
    }
    #create metadata.json file if it doesn't exist,
    # if it does, overwrite it.
    with open(base_path / "metadata.json", "w", encoding="utf8") as f:
        # write the meta dict to metadata.json file
        json.dump(meta, f, ensure_ascii=False, indent=2)


#runs the full ingestion pipeline and save three things to disk:
    #the FAISS index for similarity search
    #the embedding matrix for all text chunks
    #the metadata that describes each chunk and records which embedding model was used
if __name__ == "__main__":
    data_dir = Path("data")
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {data_dir}")

    all_chunks = []
    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem  
        title = doc_id
        print(f"Processing {pdf_path.name} as doc_id={doc_id}")
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, doc_id=doc_id, title=title)
        all_chunks.extend(chunks)

    emb, model_name = build_embeddings(all_chunks)
    index = build_faiss_index(emb)
    save_index(index, emb, all_chunks, data_dir, model_name)
    print(f"Processed {len(pdf_paths)} PDFs and created {len(all_chunks)} chunks")