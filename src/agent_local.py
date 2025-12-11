# agent_local.py

from query_local import load_resources, search, load_llm

# Simple Agentic RAG: Direct tool calling without CodeAgent framework
class SimpleAgent:
    def __init__(self):
        print("Loading Knowledge Base...")
        self.index, self.embeddings, self.chunks, self.embed_model = load_resources()
        
        print("Loading SmolLM3 Model...")
        self.llm_pipe = load_llm()
    
    def search_pdf(self, query: str) -> str:
        """Tool: Search the PDF knowledge base."""
        hits = search(query, self.index, self.chunks, self.embed_model, k=5)
        
        if not hits:
            return "No relevant documents found."
        
        result_text = "Search Results:\n"
        for i, hit in enumerate(hits, 1):
            result_text += f"\n[Result {i}] {hit['title']} (pages {hit['page_start']}-{hit['page_end']})\n"
            result_text += f"Content: {hit['text'][:500]}...\n"
        
        return result_text, hits
    
    def generate_answer(self, question: str, context_text: str) -> str:
        """Tool: Generate an answer using SmolLM3 + context."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful research assistant. Use only the provided context to answer the question. If the context is insufficient, say so clearly.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]
        
        prompt = self.llm_pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        outputs = self.llm_pipe(prompt)
        answer = outputs[0]["generated_text"][len(prompt):].strip()
        return answer
    
    def run_query(self, question: str):
        """Run an agentic query: search -> think -> generate answer."""
        print(f"\n[Agent] Received question: {question}")
        
        # Step 1: Search
        print("[Agent] Searching PDFs for relevant context...")
        search_result, hits = self.search_pdf(question)
        print(search_result)
        
        # Step 2: Generate answer using retrieved context
        if hits:
            print("[Agent] Generating answer from retrieved context...")
            context_text = "\n\n---\n\n".join([
                f"[{hit['title']}, pages {hit['page_start']}-{hit['page_end']}]\n{hit['text']}"
                for hit in hits
            ])
            answer = self.generate_answer(question, context_text)
            return answer
        else:
            return "I could not find relevant information in the documents."


# Initialize and run
print("\n" + "=" * 60)
print("Agentic Research Assistant Ready")
print("The agent will autonomously search and reason over your PDFs.")
print("=" * 60)

agent = SimpleAgent()

while True:
    try:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        if not user_input:
            continue
        
        print("\n[Agent thinking...]")
        result = agent.run_query(user_input)
        print(f"\nAnswer:\n{result}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\nGoodbye!")