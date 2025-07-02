import os
import re
from duckduckgo_search import DDGS
import google.generativeai as genai
from dotenv import load_dotenv
from autogen import AssistantAgent

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ResearchAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            system_message="You are a research assistant who provides factual, concise answers by combining real-time search with Gemini."
        )
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def duckduckgo_search(self, query, max_results=5):
        snippets = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                for i, r in enumerate(results):
                    # Include both title and body to improve factual accuracy
                    snippets.append(f"({i+1}) {r.get('title', 'No Title')}: {r.get('body', '')}")
        except Exception as e:
            snippets = [f"❌ Error during web search: {e}"]
        return snippets

    def generate_answer(self, query, snippets):
        context = "\n".join(snippets)
        prompt = f"""You are a research assistant with access to real-time web information.
Answer the following question using the given search snippets.
Cross-check conflicting information and prefer majority consensus from reputable sources.

Question: {query}

Snippets:
{context}

Answer:"""

        response = self.model.generate_content(prompt)
        return response.text.strip()

    async def run(self, query: str) -> str:
        print(f"🔍 ResearchAgent is handling query: {query}")
        snippets = self.duckduckgo_search(query)
        if not snippets:
            return "No relevant search results found."
        return self.generate_answer(query, snippets)
