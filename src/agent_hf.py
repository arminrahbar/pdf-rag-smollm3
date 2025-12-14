# src/agent_hf.py

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

# just related to cleaning up the answer returned by model.
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

# function is to help the system not print sources if answer is not
# based on context
def answer_says_no_context(answer: str) -> bool:
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

    # calls the search function defined in query_hf file
    # formats the results 
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

    # measures how close the question is to vector embeddings, meaaning to the context
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
        # searches the index for the k (in this case 5) nearest neighbor vectors to the question vector
        distances, indices = self.index.search(q_emb, k)
        # gets the distances for those 5 neighbors for this one query
        top = distances[0]
        # gets the distance to the closest neighbor
        best = float(top[0])
        # gets the mean distance across the top 5 neighbors
        mean_top = float(top.mean())
        # returns the closest distance and the mean of the 5 nearest
        return best, mean_top

    # create system systems that directs it to answer without relying on context
    # creates user message that contains only the question, without any context
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

        # sends messages to hugging face inference point
        # through an OpenAI compatible client
        completion = self.llm_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=900,
            temperature=0.3,
            top_p=0.9,
        )

        # gets the answer from the model
        raw = completion.choices[0].message.content or ""
        return strip_think(raw)

    # Decide if this question should use the PDFs (RAG) or not.
    def should_use_rag(self, question: str) -> bool:
        q_lower = question.lower()

        # indicates user wants context based answer
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
        # gets the distance of question to the nearest 5 neighbors
        # gets the mean distance of the top 5 neighbors
        # also the distance of the closest neighbor
        best, mean_top = self.corpus_distance(question, k=K_NEIGHBORS)
        print(
            f"[Router] Distances for question: best={best:.3f}, "
            f"mean_top={mean_top:.3f}"
        )

        # LOW_DISTANCE = 1.2   # "near corpus" -> RAG
        # HIGH_DISTANCE = 1.7  # "far from corpus" -> DIRECT
        # if mean distance of the 5 nearest neighbors is lower than 1.2, use RAG
        if mean_top < LOW_DISTANCE:
            print(
                f"[Router] mean distance {mean_top:.3f} < {LOW_DISTANCE}, using RAG."
            )
            return True
        
        # if mean distance of the 5 nearest neighbors is higher than 1.7, answer without RAG
        if mean_top > HIGH_DISTANCE:
            print(
                f"[Router] mean distance {mean_top:.3f} > {HIGH_DISTANCE}, "
                "using DIRECT."
            )
            return False

        # the grey zone was problematic, so all questiosn that fall in the 
        # grey zone for now are answered without using RAG. 
        # there is an area for improvement
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


    # agent’s main controller. 
    # It decides RAG vs direct, then runs the appropriate path 
    # and returns the final answer
    def run_query(self, question: str):

        print(f"\n[Agent] Received question: {question}")

        print("[Agent] Deciding whether to use RAG or direct model answer...")
        use_rag = self.should_use_rag(question)
        
        
        # three cases:
        #   Direct path: answer from model only
        #   RAG path: no chunks retrieved, 
        #   RAG path: chunks retrieved, generate answer from context
        
        
        # if use_rag is false, the agent chose the direct path
        if not use_rag:
            print("[Agent] Using direct model answer (no PDFs).")
            answer = self.generate_direct_answer(question)
            # False means the agent did not use RAG
            return answer, [], False

        # RAG path
        print("[Agent] Using RAG: searching PDFs for relevant context...")
        # Search the PDF index for nearest chunks
        search_result, hits = self.search_pdf(question)
        print(search_result)

        # if hits is empty, retrieval returned no chunks
        if not hits:
            print("[Agent] No relevant chunks found in the documents.")
            # true means the agent chose the RAG route, but retrieval failed
            return "I could not find relevant information in the documents.", [], True

        # otherwise generate the answer from the retrieved chunks
        print("[Agent] Generating answer from retrieved context...")
        raw_answer = generate_rag_answer(self.llm_client, question, hits)
        answer = strip_think(raw_answer)
        return answer, hits, True



# a runnable command line program for testing and demo.
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