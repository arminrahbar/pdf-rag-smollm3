# src/query_hf.py

from pathlib import Path
import json
import textwrap
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import torch

# Load variables from .env file (expects HF_TOKEN in there)
load_dotenv()

DATA_DIR = Path("data")
HF_API_BASE = "https://router.huggingface.co/v1"
MODEL_ID = "HuggingFaceTB/SmolLM3-3B:hf-inference"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  
K_NEIGHBORS = 5  # how many nearest chunks to use as context


def load_resources():
    """Load FAISS index and embedding model created by ingest.py."""
    index_path = DATA_DIR / "index.faiss"
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}. Run ingest.py first!")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}. Run ingest.py first!")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run ingest.py first!")

    print("Loading vector database...")
    index = faiss.read_index(str(index_path))
    embeddings = np.load(emb_path)

    with open(meta_path, "r", encoding="utf8") as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    model_name = meta["embed_model_name"]
    embed_model = SentenceTransformer(model_name)

    return index, embeddings, chunks, embed_model


def load_llm():
    """Create an OpenAI style client that talks to Hugging Face Inference."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Put HF_TOKEN=your_token in a .env file or export it in your shell."
        )

    print(f"Connecting to Hugging Face router at {HF_API_BASE} with model {MODEL_ID}")
    client = OpenAI(
        base_url=HF_API_BASE,
        api_key=token,
    )
    return client


def search(query, index, chunks, embed_model, k=K_NEIGHBORS):
    """
    Search the FAISS index for relevant chunks.
    """
    
    # creates vector for the query
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")

    # searches the index object for the nearest 5 vectors
    # to the query vector
    distances, indices = index.search(q_emb, k)

    results = []
    # iterataes through the 5 nearest neighbor indices for 
    # query vector
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        #converts from numpy integer type to regular python int
        idx = int(idx)
        # gets the chunk dictionary at current index
        chunk = chunks[idx]
        # adds this chuck to the results
        results.append(
            {
                "rank": rank + 1,
                # the distance to the query vector, convert to regular float
                "score": float(distances[0][rank]),
                # name of pdf file
                "doc_id": chunk["doc_id"],
                # name of the pdf
                "title": chunk.get("title", chunk["doc_id"]),
                # chunk's id
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": chunk["text"],
            }
        )
    return results


def generate_rag_answer(llm_client, query, retrieved_chunks):
    """Generate an answer using SmolLM3 and the retrieved context."""
    if not retrieved_chunks:
        return "I could not find any relevant context in the documents."

    # 1. Prepare context text from search results, with doc and page info
    # builds context that's made up of the chunks and their doc name, start page and end page.
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        snippet = chunk["text"][:1000]  # limit each chunk length
        doc_id = chunk["doc_id"]
        page_start = chunk["page_start"]
        page_end = chunk["page_end"]
        if page_start == page_end:
            page_str = f"{page_start}"
        else:
            page_str = f"{page_start}-{page_end}"

        context_text += (
            f"--- Excerpt {i + 1} from document {doc_id} "
            f"(pages {page_str}) ---\n"
            f"{snippet}\n\n"
        )

    
    # creates a system instruction message
    # creates a user message that includes the retrieved context and the query
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. "
                "Use only the information in the Context to answer the question. "
                "If the context is not sufficient, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\n"
                f"Question: {query}"
            ),
        },
    ]

    # sends messages to hugging face inference point
    # through an OpenAI compatible client
    completion = llm_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=900,
        temperature=0.3,
        top_p=0.9,
    )

    # gets the models response
    answer = completion.choices[0].message.content
    return answer.strip()