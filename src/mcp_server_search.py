# src/mcp_server_search.py

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from .query_hf import load_resources, search, K_NEIGHBORS

# fastMCP creates an MCP server object.  
mcp = FastMCP(
    # server name
    "Doc Search",
    transport_security=TransportSecuritySettings(
        # required so the server accepts the ngrok Host header
        enable_dns_rebinding_protection=False,
    ),
)

print("Loading RAG resources...")
index, embeddings, chunks, embed_model = load_resources()

# registers the function right below it as an MCP tool that clients can call remotely.
# when server starts, FastMCP records this function in its tool registry
@mcp.tool() 
            
# It searches your FAISS PDF index for the nearest 5 chunks matching the query
# formats each hit with doc id, title, pages, distance score, and it text   
# returns it all in JSON style dictionary. 
def search_papers(query: str, k: int = K_NEIGHBORS, max_chars: int = 1200) -> dict:
    # search function from query_hf.py
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
