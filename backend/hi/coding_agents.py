import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
from autogen import AssistantAgent

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini with your API key
genai.configure(api_key=api_key)


class CodingAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="CodingAgent",
            system_message=(
                "You are a helpful programming assistant. "
                "Always respond with clean, well-indented code wrapped in triple backticks. "
                "Only include explanations if the user asks. Default language is Python unless another is mentioned."
            )
        )
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_code(self, prompt: str, language: str = "python") -> str:
        system_prompt = f"""
You are a code generator.

Task:
Write a complete, runnable, clean {language} program that solves the following problem:

"{prompt}"

Rules:
- ONLY return the code.
- DO NOT include any explanation, markdown headings, or commentary.
- Use triple backticks and specify the language like: ```{language}
- Start the response with the code block, and end after it.

Example format:

```{language}
<your solution>
```"""

        response = self.model.generate_content(system_prompt)
        return self.extract_code(response.text, language)

    def extract_code(self, text: str, language: str = "python") -> str:
        try:
            lang_safe = re.escape(language)

            # Try to match with correct language (e.g. ```c++)
            match = re.search(rf"```{lang_safe}\s*\n(.*?)```", text, re.DOTALL)

            # Fallback: match any code block with any language tag
            if not match:
                match = re.search(r"```[\w\+\#]*\s*\n(.*?)```", text, re.DOTALL)

            # Final fallback: match a plain triple backtick block
            if not match:
                match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)

            return match.group(1).strip() if match else text.strip()

        except Exception as e:
            print(f"❌ Regex extract error: {e}")
            return text.strip()

    async def run(self, prompt: str) -> str:
        language = "python"
        for lang in ["python", "java", "c++", "c"]:
            if lang in prompt.lower():
                language = lang
                break
        print("✅ Gemini CodingAgent is now handling:", prompt)
        return self.generate_code(prompt, language)
