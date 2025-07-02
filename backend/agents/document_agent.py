import os
import base64
import PyPDF2
from langchain_core.messages import SystemMessage, AIMessage

class DocumentAgent:
    def __init__(self, groq_client):
        self.groq_client = groq_client

    async def process(self, state):
        # Initial system message
        messages = [
            {
                "role": "system",
                "content": """You are a specialized document processing agent for HPCL. 
You can analyze PDFs, extract information, summarize documents, answer questions about documents,
and process invoices and other business documents. Always provide detailed and accurate analysis."""
            }
        ]

        # Append prior conversation history from LangChain messages
        for msg in state["messages"]:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    messages.append({"role": "user", "content": msg.content})
                elif msg.__class__.__name__ == 'AIMessage':
                    messages.append({"role": "assistant", "content": msg.content})

        # 🔁 Ask Groq (or other LLM client)
        response = await self.groq_client.generate_response(messages, stream=False)

        # Return as LangChain-compatible format
        return {"messages": [AIMessage(content=response)]}

    def extract_pdf_text(self, file_path, content=None):
        try:
            # 🧠 If content is passed directly (from uploaded memory), write safely
            if content:
                with open(file_path, "wb") as f:
                    # If string, try base64 decode or fallback to utf-8 bytes
                    if isinstance(content, str):
                        try:
                            content = base64.b64decode(content)
                        except Exception:
                            content = content.encode("utf-8")
                    f.write(content)

            # 📖 Now read PDF safely
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()

        except Exception as e:
            return f"❌ Error extracting PDF text: {str(e)}"
