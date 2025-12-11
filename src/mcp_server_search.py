# src/mcp_server_search.py

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from .query_hf import load_resources, search, K_NEIGHBORS


# Create the MCP server
mcp = FastMCP(
    "Doc Search",
    transport_security=TransportSecuritySettings(
        # Required so the server accepts the ngrok Host header
        enable_dns_rebinding_protection=False,
    ),
)

print("Loading RAG resources...")
index, embeddings, chunks, embed_model = load_resources()


@mcp.tool()
def search_papers(query: str, k: int = K_NEIGHBORS, max_chars: int = 1200) -> dict:
    """
    Search the local PDF corpus using semantic similarity.

    Use this when:
    - You need information that might be contained in the uploaded biology papers.
    - The user says things like "based on the papers" or "according to these PDFs".
    - You want short excerpts plus metadata for citation.

    Arguments
    ---------
    query: Natural language question or search query.
    k: Number of nearest chunks to return.
    max_chars: Maximum characters of text to return per chunk.

    Returns
    -------
    A JSON friendly dict with:
    - query: the original query
    - k: number of neighbors requested
    - results: list of hits, each with
        rank, score, doc_id, title, page_start, page_end, page_range, text
    """
    hits = search(query, index, chunks, embed_model, k=k)

    results = []
    for h in hits:
        page_start = h["page_start"]
        page_end = h["page_end"]

        if page_start == page_end:
            page_range = str(page_start)
        else:
            page_range = f"{page_start}-{page_end}"

        text = h["text"]
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        results.append(
            {
                "rank": h["rank"],
                "score": h["score"],
                "doc_id": h["doc_id"],
                "title": h["title"],
                "page_start": page_start,
                "page_end": page_end,
                "page_range": page_range,
                "text": text,
            }
        )

    return {
        "query": query,
        "k": k,
        "results": results,
    }


if __name__ == "__main__":
    # HTTP transport so you can expose it through ngrok to ChatGPT
    mcp.run(transport="streamable-http")
