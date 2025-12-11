# langchain_query_local.py 

from pathlib import Path
import json
import textwrap
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from ingest_module import run_ingestion_pipeline

# Load variables from .env file (expects HF_TOKEN=... in there)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# CONFIGURATION
DATA_DIR = Path("data")
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_NEIGHBORS = 5  # how many nearest chunks to use as context


def load_resources():
    """Load FAISS index and embedding model, or run ingestion if files are missing."""
    index_path = DATA_DIR / "index.faiss"
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"

    # Check if index and metadata exist
    if not index_path.exists() or not meta_path.exists():
        print("Index files not found. Running full ingestion pipeline now...")
        # CALL THE NEW MASTER INGESTION FUNCTION
        return run_ingestion_pipeline(DATA_DIR)
        
    # Standard Loading (if files exist)
    print("Loading vector database...")
    index = faiss.read_index(str(index_path))
    
    with open(meta_path, "r", encoding="utf8") as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    model_name = meta["embed_model_name"]
    embed_model = SentenceTransformer(model_name)

    return index, chunks, embed_model


def load_llm():
    """Load the SmolLM3 model locally and create a text generation pipeline."""
    print(f"Loading {MODEL_ID} on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Don't set max_new_tokens here - let caller control it
        temperature=0.3,
        top_p=0.9,
    )
    return llm_pipe



def search(query, index, chunks, embed_model, k=K_NEIGHBORS):
    """
    Search the FAISS index for relevant chunks.

    Returns a list of dicts that include:
        rank, score, doc_id, page_start, page_end, text, title, chunk_id
    """
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")

    distances, indices = index.search(q_emb, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        idx = int(idx)
        chunk = chunks[idx]
        results.append(
            {
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", chunk["doc_id"]),
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": chunk["text"],
            }
        )
    return results


def generate_rag_answer(llm_pipe, query, retrieved_chunks):
    """Generate an answer using local SmolLM3 pipeline and the retrieved context."""
    if not retrieved_chunks:
        return "I could not find any relevant context in the documents."

    # 1. Prepare context text from search results, with doc and page info
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

    # 2. Build messages for chat completion
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

    # 3. Apply chat template and generate with local pipeline
    prompt = llm_pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    outputs = llm_pipe(prompt)
    generated_text = outputs[0]["generated_text"]
    
    # 4. Extract just the answer (remove the prompt from the front)
    answer = generated_text[len(prompt):]
    return answer.strip()


if __name__ == "__main__":
    # Load index and embedding model once
    index, chunks, embed_model = load_resources()
    llm_pipe = load_llm()

    print("\n" + "=" * 50)
    print(f"SmolLM3 Research Assistant Ready ({len(chunks)} chunks loaded)")
    print("=" * 50)

    while True:
        try:
            query = input("\nAsk a question (or 'q' to quit): ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                break
            if not query:
                continue

            print("Searching documents...")
            hits = search(query, index, chunks, embed_model, k=K_NEIGHBORS)

            print("Thinking...")
            answer = generate_rag_answer(llm_pipe, query, hits)

            print("\nANSWER:")
            print("-" * 20)
            print(textwrap.fill(answer, width=80))
            print("-" * 20)

            # Print sources with doc id and page numbers
            print("\nSOURCES:")
            print("-" * 20)
            for h in hits:
                if h["page_start"] == h["page_end"]:
                    page_str = f"{h['page_start']}"
                else:
                    page_str = f"{h['page_start']}-{h['page_end']}"
                print(
                    f"[{h['rank']}] {h['doc_id']} "
                    f"(pages {page_str}) "
                    f"(distance {h['score']:.4f})"
                )
            print("-" * 20)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")