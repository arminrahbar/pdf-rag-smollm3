# rag_langchain_local.py - Full LangChain Implementation with Local SmolLM3
from typing import List, Any
import re

# LangChain Imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

# Import your existing core functions and constants
from .langchain_query_local import (
    load_resources,
    load_llm,
    search,
    K_NEIGHBORS,
    MODEL_ID,
)

# Retriever to wrap Search Logic
class CustomFAISSRetriever(BaseRetriever):
    """
    A LangChain Retriever that wraps your custom FAISS search function.
    """
    index: Any = Field(default=None)
    chunks: List[Any] = Field(default_factory=list)
    embed_model: Any = Field(default=None)
    k: int = K_NEIGHBORS

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """The required method for all LangChain Retrievers."""
        hits = search(
            query,
            self.index,
            self.chunks,
            self.embed_model,
            k=self.k,
        )

        docs = []
        for hit in hits:
            metadata = {
                "doc_id": hit["doc_id"],
                "title": hit["title"],
                "page_start": hit["page_start"],
                "page_end": hit["page_end"],
                "distance_score": hit["score"],
            }
            docs.append(Document(page_content=hit["text"], metadata=metadata))

        return docs


def _format_docs(context_docs: List[Document]) -> str:
    """Helper function to format documents for the prompt context."""
    context_text = ""
    for i, doc in enumerate(context_docs):
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        if page_start == page_end:
            page_str = f"{page_start}"
        else:
            page_str = f"{page_start}-{page_end}"

        context_text += (
            f"--- Excerpt {i + 1} from document {doc.metadata.get('doc_id')} "
            f"(pages {page_str}) ---\n"
            f"{doc.page_content[:1000]}\n\n"
        )
    return context_text


# Distance-Based Router
class DistanceRouter:
    """Routes queries based on embedding distance to corpus."""

    def __init__(self, index, chunks, embed_model):
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model
        self.LOW_DISTANCE = 1.2
        self.HIGH_DISTANCE = 1.7

    def compute_distance(self, question: str, k: int = K_NEIGHBORS) -> float:
        """Compute mean distance of question to top k corpus chunks."""
        q_emb = self.embed_model.encode(
            [question],
            convert_to_numpy=True,
        ).astype("float32")
        distances, _ = self.index.search(q_emb, k)
        mean_distance = float(distances[0].mean())
        return mean_distance

    def should_use_rag(self, question: str) -> bool:
        """Decide if RAG should be used based on distance."""
        q_lower = question.lower()

        # Explicit PDF mentions always use RAG
        explicit_phrases = [
            "based on the pdf",
            "based on the paper",
            "according to the pdf",
            "in this pdf",
            "in this paper",
            "in these pdfs",
        ]
        if any(phrase in q_lower for phrase in explicit_phrases):
            print("[Router] User explicitly mentioned PDFs -> RAG")
            return True

        # Distance based routing
        mean_dist = self.compute_distance(question)
        print(f"[Router] Mean distance to corpus: {mean_dist:.3f}")

        if mean_dist < self.LOW_DISTANCE:
            print(f"[Router] Distance {mean_dist:.3f} < {self.LOW_DISTANCE} -> RAG")
            return True
        elif mean_dist > self.HIGH_DISTANCE:
            print(f"[Router] Distance {mean_dist:.3f} > {self.HIGH_DISTANCE} -> DIRECT")
            return False
        else:
            print(f"[Router] Grey zone ({mean_dist:.3f}), defaulting to DIRECT")
            return False


# Custom Output Parser to Clean Model Response
class CleanOutputParser(StrOutputParser):
    """Extracts only the answer portion from the model's output."""

    def detect_repetition(self, text: str) -> int:
        """Detect where repetition starts. Returns index or -1."""
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if sentences[i] == sentences[j]:
                    repeat_start = text.find(
                        sentences[j],
                        text.find(sentences[i]) + len(sentences[i]),
                    )
                    return repeat_start
        return -1

    def parse(self, text: str) -> str:
        # Remove <think>...</think> tags
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # If there is an "Answer:" marker, extract everything after it
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            answer = text.strip()

        return answer


# Create Routed Chain (RAG vs Direct) for local pipeline
def create_routed_chain(llm_pipe, retriever, router):
    """Creates a branching chain that routes between RAG and direct answer."""

    # Ensure tokenizer has pad_token set
    if llm_pipe.tokenizer.pad_token is None:
        llm_pipe.tokenizer.pad_token = llm_pipe.tokenizer.eos_token

    # LLM wrapper for local Hugging Face pipeline, using chat template
    class DynamicLLM:
        def __init__(self, pipeline):
            self.pipeline = pipeline
            self.tokenizer = pipeline.tokenizer

        def _generate(self, prompt_text: str) -> str:
            # Wrap the prompt in a chat template so the model sees proper chat format
            messages = [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ]

            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            result = self.pipeline(
                chat_prompt,
                max_new_tokens=900,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                return_full_text=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated = result[0]["generated_text"]
            return generated

        def __call__(self, prompt: str) -> str:
            return self._generate(prompt)

        def invoke(self, input_data):
            if isinstance(input_data, str):
                prompt_text = input_data
            elif hasattr(input_data, "text"):
                prompt_text = input_data.text
            else:
                prompt_text = str(input_data)
            return self._generate(prompt_text)

    llm = DynamicLLM(llm_pipe)

    # RAG prompt
    rag_prompt_template = """You are a helpful research assistant. Use only the information in the Context to answer the question. If the context is not sufficient, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    rag_prompt = PromptTemplate(
        template=rag_prompt_template,
        input_variables=["context", "question"],
    )

    # Direct answer prompt
    direct_prompt_template = """You are a helpful research assistant. Answer from your general knowledge.

Question: {question}

Answer:"""

    direct_prompt = PromptTemplate(
        template=direct_prompt_template,
        input_variables=["question"],
    )

    # Custom parser to clean output
    output_parser = CleanOutputParser()

    # Helper to format prompt and call LLM
    def call_llm_with_prompt(prompt_template, input_dict):
        formatted = prompt_template.format(**input_dict)
        response = llm(formatted)
        return output_parser.parse(response)

    # RAG chain
    def rag_chain_func(input_dict):
        context_docs = retriever.invoke(input_dict["question"])
        context_text = _format_docs(context_docs)

        return call_llm_with_prompt(
            rag_prompt,
            {
                "context": context_text,
                "question": input_dict["question"],
            },
        )

    # Direct chain
    def direct_chain_func(input_dict):
        return call_llm_with_prompt(
            direct_prompt,
            {
                "question": input_dict["question"],
            },
        )

    # Routing function
    def full_chain(input_dict):
        if router.should_use_rag(input_dict["question"]):
            return rag_chain_func(input_dict)
        else:
            return direct_chain_func(input_dict)

    return full_chain


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Loading resources...")
    print("=" * 60)

    # Load local pipeline and vector resources
    llm_pipe = load_llm()
    index, chunks, embed_model = load_resources()

    # Create components
    retriever = CustomFAISSRetriever(
        index=index,
        chunks=chunks,
        embed_model=embed_model,
    )

    router = DistanceRouter(index, chunks, embed_model)

    # Create the full routed chain
    chain = create_routed_chain(llm_pipe, retriever, router)

    print("\n" + "=" * 60)
    print("LangChain RAG Router Ready (Local SmolLM3)")
    print("Distance based routing: Close to corpus -> RAG, Far -> Direct")
    print("=" * 60)

    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                break
            if not query:
                continue

            print("\n[Routing...]")

            answer = chain({"question": query})

            print("\nAnswer:")
            print("-" * 20)
            print(answer)
            print("-" * 20)
            print("Answer length:", len(answer))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    print("\nGoodbye!")