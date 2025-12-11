# agent_hf.py

from pathlib import Path
import textwrap
import re

from .query_hf import (
    load_resources,
    load_llm,
    search,
    generate_rag_answer,
    K_NEIGHBORS,
    MODEL_ID,
)

# Distance thresholds for routing
# You will want to calibrate these by printing distances for various questions.
LOW_DISTANCE = 1.2   # "near corpus" -> RAG
HIGH_DISTANCE = 1.7  # "far from corpus" -> DIRECT

# Whether to use the LLM as a router in the grey zone
USE_LLM_ROUTER = False


def strip_think(text: str) -> str:
    """
    Remove any <think>...</think> segments from model output, even if truncated.
    """
    if not text:
        return ""
    # Remove from <think> up to </think> or end of string
    return re.sub(
        r"<think>.*?(</think>|$)", "", text, flags=re.IGNORECASE | re.DOTALL
    ).strip()


def answer_says_no_context(answer: str) -> bool:
    """
    Heuristic detector: does the RAG answer explicitly say that
    the documents or context do not cover the question.
    """
    text = answer.lower()
    patterns = [
        "the provided documents do not",
        "the provided context does not",
        "do not discuss",
        "does not discuss",
        "do not address",
        "does not address",
        "no information about",
        "not possible to determine",
        "cannot determine from the provided context",
        "context is not sufficient",
        "based on the given context, it is not possible",
    ]
    return any(p in text for p in patterns)


class SimpleAgent:
    def __init__(self):
        print("Loading Knowledge Base...")
        self.index, self.embeddings, self.chunks, self.embed_model = load_resources()

        print("Connecting to SmolLM3 on Hugging Face...")
        self.llm_client = load_llm()

    def search_pdf(self, query: str):
        """Tool: Search the PDF knowledge base."""
        hits = search(query, self.index, self.chunks, self.embed_model, k=K_NEIGHBORS)

        if not hits:
            return "No relevant documents found.", []

        result_text = "Search Results:\n"
        for i, hit in enumerate(hits, 1):
            page_start = hit["page_start"]
            page_end = hit["page_end"]
            if page_start == page_end:
                page_str = f"{page_start}"
            else:
                page_str = f"{page_start}-{page_end}"

            result_text += (
                f"\n[Result {i}] {hit['title']} (pages {page_str})\n"
                f"Content: {hit['text'][:500]}...\n"
            )

        return result_text, hits

    def corpus_distance(self, question: str, k: int = 5):
        """
        Compute how close the question is to the existing corpus in embedding space.

        Returns:
            best_distance (float),
            mean_top_k_distance (float)
        """
        q_emb = self.embed_model.encode([question], convert_to_numpy=True).astype(
            "float32"
        )
        distances, indices = self.index.search(q_emb, k)
        top = distances[0]
        best = float(top[0])
        mean_top = float(top.mean())
        return best, mean_top

    def generate_direct_answer(self, question: str) -> str:
        """Answer using the model's own knowledge only, no PDFs."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. Answer from your general "
                    "knowledge. Do not mention any PDFs or external tools."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        completion = self.llm_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=900,
            temperature=0.3,
            top_p=0.9,
        )

        raw = completion.choices[0].message.content or ""
        return strip_think(raw)

    # def should_use_rag_via_llm(self, question: str) -> bool:
    #     """
    #     Grey zone router: ask the LLM to choose DIRECT vs RAG.

    #     Returns True for RAG, False for DIRECT.
    #     """
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": (
    #                 "You are a router for a research assistant.\n"
    #                 "Decide if this question requires consulting the uploaded PDF "
    #                 "documents.\n"
    #                 "Answer DIRECT if you can answer confidently from your own general "
    #                 "knowledge without reading the PDFs.\n"
    #                 "Answer RAG only if the question clearly depends on the specific "
    #                 "uploaded articles, their figures, tables, methods, or detailed "
    #                 "results, or explicitly refers to 'this paper', 'these PDFs', or "
    #                 "their scientific findings.\n"
    #                 "For generic background questions (for example 'what is biology', "
    #                 "'what is data science', 'what is a protein'), prefer DIRECT.\n"
    #                 "Reply with exactly one word: DIRECT or RAG."
    #             ),
    #         },
    #         {"role": "user", "content": question},
    #     ]

    #     completion = self.llm_client.chat.completions.create(
    #         model=MODEL_ID,
    #         messages=messages,
    #         max_tokens=16,
    #         temperature=0.0,
    #     )

    #     reply = completion.choices[0].message.content or ""
    #     reply_clean = strip_think(reply).strip().upper()

    #     print(f"[Router raw reply] {repr(reply_clean)}")

    #     # Clear cases
    #     if "RAG" in reply_clean and "DIRECT" not in reply_clean:
    #         return True
    #     if "DIRECT" in reply_clean and "RAG" not in reply_clean:
    #         return False

    #     # Ambiguous or nonsense: default to DIRECT to avoid overusing RAG
    #     print("[Router] Ambiguous router reply, defaulting to DIRECT.")
    #     return False

    def should_use_rag(self, question: str) -> bool:
        """
        Decide if this question should use the PDFs (RAG) or not.

        Logic:
          1) If user explicitly asks to use PDFs or papers, force RAG.
          2) Otherwise use embedding distance to corpus.
          3) If in grey zone and USE_LLM_ROUTER is True, ask LLM router as tie breaker.
        """
        q_lower = question.lower()

        # 1. Explicit user request to ground in PDFs
        explicit_pdf_phrases = [
            "based on the pdf",
            "based on the pdfs",
            "based on the paper",
            "based on the papers",
            "according to the pdf",
            "according to the paper",
            "according to these pdfs",
            "according to these papers",
            "in this pdf",
            "in this paper",
            "in these pdfs",
            "in these papers",
        ]
        if any(phrase in q_lower for phrase in explicit_pdf_phrases):
            print("[Router] User explicitly asked to use PDFs -> RAG.")
            return True

        # 2. Distance based routing
        best, mean_top = self.corpus_distance(question, k=K_NEIGHBORS)
        print(
            f"[Router] Distances for question: best={best:.3f}, "
            f"mean_top={mean_top:.3f}"
        )

        if mean_top < LOW_DISTANCE:
            print(
                f"[Router] mean distance {mean_top:.3f} < {LOW_DISTANCE}, using RAG."
            )
            return True

        if mean_top > HIGH_DISTANCE:
            print(
                f"[Router] mean distance {mean_top:.3f} > {HIGH_DISTANCE}, "
                "using DIRECT."
            )
            return False

        # 3. Grey zone
        if not USE_LLM_ROUTER:
            print(
                f"[Router] Grey zone (mean distance {mean_top:.3f}), "
                "no LLM router configured, using DIRECT."
            )
            return False

        print(
            f"[Router] Grey zone (mean distance {mean_top:.3f}), "
            "asking LLM router."
        )
        return self.should_use_rag_via_llm(question)

    def run_query(self, question: str):
        """
        Run a routed query:

          - If router says DIRECT: answer from model only, no PDFs.
          - If router says RAG: search PDFs and answer with retrieved context.

        Returns:
            answer (str),
            hits (list of chunks, or [] if no RAG),
            used_rag (bool)
        """
        print(f"\n[Agent] Received question: {question}")

        print("[Agent] Deciding whether to use RAG or direct model answer...")
        use_rag = self.should_use_rag(question)

        if not use_rag:
            print("[Agent] Using direct model answer (no PDFs).")
            answer = self.generate_direct_answer(question)
            return answer, [], False

        # RAG path
        print("[Agent] Using RAG: searching PDFs for relevant context...")
        search_result, hits = self.search_pdf(question)
        print(search_result)

        if not hits:
            print("[Agent] No relevant chunks found in the documents.")
            return "I could not find relevant information in the documents.", [], True

        print("[Agent] Generating answer from retrieved context...")
        raw_answer = generate_rag_answer(self.llm_client, question, hits)
        answer = strip_think(raw_answer)
        return answer, hits, True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Agentic Research Assistant Ready (HF SmolLM3 + RAG router)")
    print("The agent will decide when to use PDFs and when to answer directly.")
    print("=" * 60)

    agent = SimpleAgent()

    while True:
        try:
            user_input = input("\nYour question: ").strip()
            if user_input.lower() in ["q", "quit", "exit"]:
                break
            if not user_input:
                continue

            print("\n[Agent thinking...]")
            answer, hits, used_rag = agent.run_query(user_input)

            print("\nAnswer:")
            print("-" * 20)
            print(textwrap.fill(answer, width=80))
            print("-" * 20)

            print("\nSOURCES:")
            print("-" * 20)

            # Detect the special case: RAG was used, but the answer says
            # the documents do not cover this question.
            context_not_covered = used_rag and answer_says_no_context(answer)

            if not used_rag:
                print("Answered from model's general knowledge only (no PDFs used).")
            else:
                if context_not_covered:
                    print("The documents do not address this question, so no sources are listed.")
                elif not hits:
                    print("No relevant chunks found in the documents.")
                else:
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
            import traceback

            traceback.print_exc()

    print("\nGoodbye!")