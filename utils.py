import os
import json
import tempfile
from pathlib import Path

from groq import Groq

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_api_key():
    """Get the Groq API key from Streamlit Secrets or an environment variable."""
    try:
        import streamlit as st

        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


def get_groq_client():
    """Create a Groq client."""
    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT Streamlit uploads."""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if filename.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    if filename.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed.")

        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if not text:
            raise ValueError(
                "No selectable text was found in this PDF. "
                "Scanned/image-only PDFs require OCR."
            )

        return text

    if filename.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx is not installed.")

        with tempfile.NamedTemporaryFile(
            suffix=".docx", delete=False
        ) as temp:
            temp.write(data)
            temp_path = temp.name

        try:
            document = Document(temp_path)
            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ).strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    raise ValueError(
        "Unsupported file type. Please upload PDF, DOCX, or TXT."
    )


def parse_json_response(text):
    """Parse JSON and tolerate accidental Markdown code fences."""
    if not text:
        raise ValueError("The AI returned an empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise ValueError("The AI did not return valid JSON.")

        return json.loads(cleaned[start:end + 1])


def call_groq(prompt, json_output=False):
    """
    Central Groq API wrapper used by all five workflow stages.
    """
    client = get_groq_client()

    request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert educational AI assistant. "
                    "Follow the user's instructions precisely. "
                    "Prioritize factual accuracy, clarity, and useful learning."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3 if json_output else 0.45,
        "max_tokens": 7000,
    }

    if json_output:
        request["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    if not response.choices:
        raise RuntimeError("Groq returned no choices.")

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Groq returned an empty response.")

    if json_output:
        return parse_json_response(content)

    return content.strip()
