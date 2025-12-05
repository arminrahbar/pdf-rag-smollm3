from pathlib import Path
import json

import numpy as np
import faiss # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore



DATA_DIR = Path("data")


def load_resources():
    # Load FAISS index
    index_path = DATA_DIR / "index.faiss"
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    index = faiss.read_index(str(index_path))
    embeddings = np.load(emb_path)

    with open(meta_path, "r", encoding="utf8") as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    model_name = meta["embed_model_name"]
    model = SentenceTransformer(model_name)

    return index, embeddings, chunks, model


def search(query, k=5):
    index, embeddings, chunks, model = load_resources()

    # Encode query
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")

    # Search
    distances, indices = index.search(q_emb, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        idx = int(idx)
        chunk = chunks[idx]
        results.append(
            {
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text_preview": chunk["text"][:300],
            }
        )
    return results


if __name__ == "__main__":
    import textwrap

    query = input("Enter your question: ").strip()
    hits = search(query, k=5)

    for hit in hits:
        print()
        print(f"Result {hit['rank']} (distance {hit['score']:.4f})")
        print(f"Document: {hit['doc_id']} | pages {hit['page_start']} to {hit['page_end']}")
        print(textwrap.shorten(hit["text_preview"], width=300, placeholder="..."))
