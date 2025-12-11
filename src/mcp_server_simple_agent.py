# src/mcp_server_simple_agent.py

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from .agent_hf import SimpleAgent, answer_says_no_context

mcp = FastMCP(
    "PDF Research Assistant",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)

print("Loading agent and RAG resources...")
agent = SimpleAgent()  # This loads FAISS and connects to SmolLM3 on HF


@mcp.tool()
def ask_papers(question: str) -> dict:
    """
    Use the SmolLM3 + RAG agent to answer a question.

    Returns a JSON friendly dict with the answer and any sources.
    """
    answer, hits, used_rag = agent.run_query(question)
    context_not_covered = used_rag and answer_says_no_context(answer)

    # Decide whether to include sources (Messi case: no sources)
    if not used_rag or context_not_covered or not hits:
        sources = []
    else:
        sources = []
        for h in hits:
            if h["page_start"] == h["page_end"]:
                page_str = f"{h['page_start']}"
            else:
                page_str = f"{h['page_start']}-{h['page_end']}"

            sources.append(
                {
                    "rank": h["rank"],
                    "doc_id": h["doc_id"],
                    "title": h["title"],
                    "page_start": h["page_start"],
                    "page_end": h["page_end"],
                    "page_str": page_str,
                    "distance": h["score"],
                }
            )

    return {
        "answer": answer,
        "used_rag": used_rag,
        "context_not_covered": context_not_covered,
        "sources": sources,
    }


if __name__ == "__main__":
    # Use Streamable HTTP transport that ChatGPT can connect to
    mcp.run(transport="streamable-http")