import os, json, tempfile
from pathlib import Path
from google import genai
from google.genai import types
try: from pypdf import PdfReader
except ImportError: PdfReader = None
try: from docx import Document
except ImportError: Document = None
DEFAULT_MODEL=os.getenv('GEMINI_MODEL','gemini-3.7-flash')
def get_api_key():
    try:
        import streamlit as st
        key=st.secrets.get('GEMINI_API_KEY')
        if key: return str(key).strip()
    except Exception: pass
    return os.getenv('GEMINI_API_KEY','').strip()
def get_gemini_client():
    key=get_api_key()
    if not key: raise ValueError('GEMINI_API_KEY is missing. Add it to Streamlit Secrets.')
    return genai.Client(api_key=key)
def extract_text_from_file(f):
    if f is None: return ''
    name=f.name.lower(); data=f.getvalue()
    if name.endswith('.txt'): return data.decode('utf-8',errors='ignore').strip()
    if name.endswith('.pdf'):
        if PdfReader is None: raise RuntimeError('pypdf is not installed.')
        from io import BytesIO
        text='\n\n'.join(p.extract_text() or '' for p in PdfReader(BytesIO(data)).pages).strip()
        if not text: raise ValueError('No selectable text found. Scanned PDFs need OCR.')
        return text
    if name.endswith('.docx'):
        if Document is None: raise RuntimeError('python-docx is not installed.')
        with tempfile.NamedTemporaryFile(suffix='.docx',delete=False) as t: t.write(data); path=t.name
        try: return '\n'.join(p.text for p in Document(path).paragraphs if p.text.strip()).strip()
        finally: Path(path).unlink(missing_ok=True)
    raise ValueError('Unsupported file type. Please upload PDF, DOCX, or TXT.')
def parse_json_response(text):
    s=(text or '').strip()
    if s.startswith('```'):
        lines=s.splitlines(); lines=lines[1:]; lines=lines[:-1] if lines and lines[-1].strip()=='```' else lines; s='\n'.join(lines).strip()
    try: return json.loads(s)
    except json.JSONDecodeError:
        a,b=s.find('{'),s.rfind('}')
        if a<0 or b<=a: raise ValueError('Gemini did not return valid JSON.')
        return json.loads(s[a:b+1])
def call_gemini(prompt,json_output=False):
    config=types.GenerateContentConfig(temperature=.3 if json_output else .45,max_output_tokens=7000)
    if json_output: config.response_mime_type='application/json'
    r=get_gemini_client().models.generate_content(model=DEFAULT_MODEL,contents=prompt,config=config)
    if not r.text: raise RuntimeError('Gemini returned an empty response.')
    return parse_json_response(r.text) if json_output else r.text.strip()
