# rag_langchain_hf.py - Full LangChain Implementation with HuggingFace Inference API
from typing import List, Any
import textwrap
import numpy as np
import re

# LangChain Imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Import existing core functions and constants
from .langchain_query_hf import (
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
            k=self.k
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


def _format_docs(context_docs):
    """Helper function to format documents for the prompt context."""
    context_text = ""
    for i, doc in enumerate(context_docs):
        page_start = doc.metadata.get('page_start')
        page_end = doc.metadata.get('page_end')
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
    
    def compute_distance(self, question: str, k: int = K_NEIGHBORS):
        """Compute mean distance of question to top-k corpus chunks."""
        q_emb = self.embed_model.encode([question], convert_to_numpy=True).astype("float32")
        distances, _ = self.index.search(q_emb, k)
        mean_distance = float(distances[0].mean())
        return mean_distance
    
    def should_use_rag(self, question: str) -> bool:
        """Decide if RAG should be used based on distance."""
        q_lower = question.lower()
        
        # Explicit PDF mentions always use RAG
        explicit_phrases = [
            "based on the pdf", "based on the paper", "according to the pdf",
            "in this pdf", "in this paper", "in these pdfs"
        ]
        if any(phrase in q_lower for phrase in explicit_phrases):
            print("[Router] User explicitly mentioned PDFs -> RAG")
            return True
        
        # Distance-based routing
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
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                # If we find the same sentence later, cut before the repeat
                if sentences[i] == sentences[j]:
                    # Find where this sentence starts in the original text
                    repeat_start = text.find(sentences[j], text.find(sentences[i]) + len(sentences[i]))
                    return repeat_start
        return -1
    
    def parse(self, text: str) -> str:
        import re
        
        # Remove <think>...</think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
        
        # If there's an "Answer:" marker, extract everything after it
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            answer = text.strip()
        
        return answer
    

# Create Routed Chain (RAG vs Direct)
def create_routed_chain(llm_client, retriever, router):
    """Creates a branching chain that routes between RAG and direct answer."""
    
    # Create LLM wrapper for HuggingFace Inference API
    class DynamicLLM:
        def __init__(self, client):
            self.client = client

        def __call__(self, prompt: str) -> str:
            completion = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
                top_p=0.9,
            )
            return completion.choices[0].message.content

        def invoke(self, input_data):
            if isinstance(input_data, str):
                return self.__call__(input_data)
            else:
                return self.__call__(str(input_data))
    
    llm = DynamicLLM(llm_client)
    
    # RAG prompt
    rag_prompt_template = """You are a helpful research assistant. Use only the information in the Context to answer the question. If the context is not sufficient, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    
    rag_prompt = PromptTemplate(
        template=rag_prompt_template,
        input_variables=["context", "question"]
    )
    
    # Direct answer prompt
    direct_prompt_template = """You are a helpful research assistant. Answer from your general knowledge.

Question: {question}

Answer:"""
    
    direct_prompt = PromptTemplate(
        template=direct_prompt_template,
        input_variables=["question"]
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
        # Get context from retriever
        context_docs = retriever.invoke(input_dict['question'])
        context_text = _format_docs(context_docs)
        
        # Format and call
        return call_llm_with_prompt(rag_prompt, {
            'context': context_text,
            'question': input_dict['question']
        })
    
    # Direct chain
    def direct_chain_func(input_dict):
        return call_llm_with_prompt(direct_prompt, {
            'question': input_dict['question']
        })
    
    rag_chain = rag_chain_func
    direct_chain = direct_chain_func
    
    # Routing function
    def full_chain(input_dict):
        if router.should_use_rag(input_dict["question"]):
            return rag_chain(input_dict)
        else:
            return direct_chain(input_dict)
    
    return full_chain

def print_sources_from_docs(docs):
    print("\nSOURCES:")
    print("-" * 20)
    if not docs:
        print("No relevant chunks found in the documents.")
        print("-" * 20)
        return

    for i, doc in enumerate(docs, start=1):
        doc_id = doc.metadata.get("doc_id", "unknown_doc")
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        distance = doc.metadata.get("distance_score", None)

        if page_start == page_end:
            page_str = f"{page_start}"
        else:
            page_str = f"{page_start}-{page_end}"

        if distance is not None:
            print(
                f"[{i}] {doc_id} "
                f"(pages {page_str}) "
                f"(distance {distance:.4f})"
            )
        else:
            print(f"[{i}] {doc_id} (pages {page_str})")
    print("-" * 20)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Loading resources...")
    print("=" * 60)
    
    # Load all resources
    llm_client = load_llm()
    index, chunks, embed_model = load_resources()
    
    # Create components
    retriever = CustomFAISSRetriever(
        index=index, chunks=chunks, embed_model=embed_model
    )
    
    router = DistanceRouter(index, chunks, embed_model)
    
    # Create the full routed chain
    chain = create_routed_chain(llm_client, retriever, router)

    print("\n" + "=" * 60)
    print("LangChain RAG Router Ready (HuggingFace Inference API)")
    print("Distance-based routing: Close to corpus -> RAG, Far -> Direct")
    print("=" * 60)

    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                break
            if not query:
                continue

            print("\n[Routing...]")

            # Decide routing here as well so you know whether RAG was used
            use_rag = router.should_use_rag(query)

            # Call the routed chain (still does its own routing)
            answer = chain({"question": query})

            print("\nAnswer:")
            print("-" * 20)
            print(answer)
            print("-" * 20)
            print("Answer length:", len(answer))

            # Print sources in the same style as the non LangChain version
            if not use_rag:
                print("\nSOURCES:")
                print("-" * 20)
                print("Answered from model's general knowledge only (no PDFs used).")
                print("-" * 20)
            else:
                # Retrieve the same top-k docs and print their doc_id and page range
                context_docs = retriever.invoke(query)
                print_sources_from_docs(context_docs)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nGoodbye!")