import os
import re
import io
import base64
import black
import pandas as pd
import plotly.express as px
import plotly.io as pio
import google.generativeai as genai
from dotenv import load_dotenv
from autogen import AssistantAgent
from contextlib import redirect_stdout
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class AnalyticsAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="AnalyticsAgent",
            system_message="You are a data analytics expert who writes Python code using pandas and Plotly."
        )
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def extract_code(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:python)?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.lstrip("`").lstrip().replace("python", "", 1).lstrip()
        return cleaned.strip()

    def generate_code(self, df_sample: pd.DataFrame, user_prompt: str) -> str:
        prompt = f"""
You are a data analyst working with pandas and Plotly in Python.

Do not read or write any external files (like 'data.csv').
Assume a pandas DataFrame named `df` is already loaded with the following sample data:

{df_sample.to_csv(index=False)}

Instruction: {user_prompt}

Return only the valid Python code wrapped in triple backticks. 
Use actual column names from the DataFrame.
Avoid using placeholder names like 'x_column' or 'data.csv'.
"""
        response = self.model.generate_content(prompt)
        raw_code = self.extract_code(response.text)
        raw_code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", raw_code)
        try:
            return black.format_str(raw_code, mode=black.Mode())
        except Exception:
            return raw_code

    async def run(self, file: dict, user_prompt: str) -> str:
        filename = file["name"]
        content = file["content"]
        code = ""  # Always define `code` early

        try:
            # Handle file types
            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(content))

            elif filename.endswith(".xlsx"):
                try:
                    if isinstance(content, bytes):
                        content_bytes = content
                    elif isinstance(content, str):
                        try:
                            content_bytes = base64.b64decode(content)
                        except Exception:
                            return "❌ Could not decode base64 string for .xlsx file"
                    else:
                        return "❌ Unsupported content type for Excel file."

                    df = pd.read_excel(io.BytesIO(content_bytes), engine="openpyxl")
                except Exception as e:
                    return f"❌ Failed to load Excel file:\n\n{e}"

            elif filename.endswith(".pdf"):
                os.makedirs("temp", exist_ok=True)
                temp_path = os.path.join("temp", filename)
                with open(temp_path, "wb") as f:
                    f.write(content.encode())
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                combined_text = "\n".join([page.page_content for page in pages])
                return f"📄 PDF content received:\n\n{combined_text[:3000]}"

            else:
                return "❌ Unsupported file format. Upload .csv, .xlsx, or .pdf."

            # Generate code
            code = self.generate_code(df.head(5), user_prompt)
            local_vars = {"pd": pd, "px": px, "df": df}
            buffer = io.StringIO()

            try:
                with redirect_stdout(buffer):
                    exec(code, local_vars)
            except SystemExit:
                return "❌ Your code attempted to exit the app using `exit()` or `SystemExit`, which has been blocked."
            except Exception as e:
                return f"❌ An error occurred while executing the code:\n\n{e}\n\n```python\n{code}\n```"

            # Return generated figure if available
            fig = None
            for val in local_vars.values():
                if isinstance(val, px.Figure) or hasattr(val, "to_html"):
                    fig = val
                    break

            if fig:
                html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
                return html
            else:
                return f"✅ Code executed but no plot was returned.\n\n```python\n{code}\n```"

        except Exception as e:
            return f"❌ Error executing code:\n\n{e}\n\n```python\n{code}\n```"
