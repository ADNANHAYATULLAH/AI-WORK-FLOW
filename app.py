import os
import json
import tempfile
from pathlib import Path

import gradio as gr
import streamlit as st
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


APP_TITLE = "AI Study Pack Generator"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

STAGES = [
    "Planning",
    "Content Generation",
    "Assessment",
    "Review",
    "Refinement",
]


# -----------------------------
# 1. API + FILE HELPERS
# -----------------------------

def get_api_key():
    """Read the Gemini key from Streamlit Secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()


def get_client():
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )
    return genai.Client(api_key=api_key)


def extract_text_from_file(uploaded_file):
    """Extract text from TXT, PDF, or DOCX."""
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed.")
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx is not installed.")

        with tempfile.NamedTemporaryFile(
            suffix=".docx", delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            document = Document(tmp_path)
            return "\n".join(p.text for p in document.paragraphs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def clean_json(text):
    """Make a best effort to extract JSON from a Gemini response."""
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def call_gemini(prompt, expect_json=False):
    """Single AI gateway used by every workflow stage."""
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=0.3 if expect_json else 0.45,
        max_output_tokens=7000,
    )

    if expect_json:
        config.response_mime_type = "application/json"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    if expect_json:
        return clean_json(response.text)

    return response.text.strip()


# -----------------------------
# 2. WORKFLOW STAGES
# -----------------------------

def stage_planning(source_text, topic, level, difficulty, language, sections):
    prompt = f"""
You are Stage 1 of a multi-stage AI study-pack workflow.

ROLE: Expert instructional planner.

Create a study-pack PLAN before any content is written.

Student profile:
- Level: {level}
- Difficulty: {difficulty}
- Language: {language}
- Requested sections: {", ".join(sections)}

Topic:
{topic or "Infer the topic from the source material."}

Source material:
{source_text[:24000]}

Return ONLY valid JSON with exactly these keys:
{{
  "title": "string",
  "topic": "string",
  "learning_objectives": ["string"],
  "key_concepts": ["string"],
  "content_sections": ["string"],
  "question_count": 8,
  "study_plan_days": 5,
  "difficulty_notes": "string"
}}

Planning rules:
- Objectives must be measurable.
- Select concepts that are actually supported by the source.
- Keep the plan appropriate for the student's level.
- Do not write the final study pack yet.
"""
    return call_gemini(prompt, expect_json=True)


def stage_content(source_text, user_topic, profile, plan):
    prompt = f"""
You are Stage 2 of a multi-stage AI study-pack workflow.

ROLE: Expert teacher and educational content writer.

Generate the first complete study pack using the PLAN below.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

PLAN FROM STAGE 1:
{json.dumps(plan, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source_text[:24000]}

USER TOPIC:
{user_topic or "Use the topic identified by the planner."}

Create:
1. A concise topic summary.
2. Key concepts with explanations.
3. Flashcards.
4. Multiple-choice questions with 4 options.
5. A short practice quiz.
6. A practical study plan.

Use Markdown.
Use only information supported by the source when source material is supplied.
If the source does not contain enough information, clearly label general knowledge
rather than pretending it came from the source.

Do not discuss the workflow. Return only the draft study pack.
"""
    return call_gemini(prompt)


def stage_assessment(draft, profile, plan):
    prompt = f"""
You are Stage 3 of a multi-stage AI study-pack workflow.

ROLE: Assessment designer and quality checker.

Evaluate the draft study pack against the plan.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

DRAFT:
{draft[:30000]}

Return ONLY valid JSON:
{{
  "accuracy_score": 0,
  "coverage_score": 0,
  "difficulty_score": 0,
  "clarity_score": 0,
  "assessment_score": 0,
  "strengths": ["string"],
  "issues": ["string"],
  "missing_topics": ["string"],
  "question_issues": ["string"],
  "recommended_fixes": ["string"]
}}

Scores must be integers from 0 to 100.

Check:
- factual consistency
- coverage of planned objectives
- appropriate difficulty
- clarity
- usefulness of questions
- answer quality
"""
    return call_gemini(prompt, expect_json=True)


def stage_review(draft, assessment, source_text):
    prompt = f"""
You are Stage 4 of a multi-stage AI study-pack workflow.

ROLE: Senior reviewer.

Review the draft and the assessment results before the final rewrite.

DRAFT:
{draft[:30000]}

ASSESSMENT:
{json.dumps(assessment, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source_text[:18000]}

Return ONLY valid JSON:
{{
  "pass": true,
  "priority_fixes": ["string"],
  "source_conflicts": ["string"],
  "format_fixes": ["string"],
  "final_instructions": ["string"]
}}

Set pass=true only if the pack is already strong enough to publish.
Be strict but practical. Focus on fixes that materially improve the result.
"""
    return call_gemini(prompt, expect_json=True)


def stage_refinement(draft, plan, assessment, review, profile, source_text):
    prompt = f"""
You are Stage 5 of a multi-stage AI study-pack workflow.

ROLE: Final editor.

Rewrite the draft into the FINAL study pack.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

ASSESSMENT:
{json.dumps(assessment, ensure_ascii=False, indent=2)}

REVIEW:
{json.dumps(review, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source_text[:22000]}

DRAFT:
{draft[:30000]}

Apply the review fixes. Preserve correct content while improving weak areas.

Final output must contain:
# Study Pack Title
## 1. Learning Objectives
## 2. Topic Summary
## 3. Key Concepts
## 4. Flashcards
## 5. Multiple-Choice Questions
## 6. Practice Quiz
## 7. Study Plan
## 8. Quick Revision Checklist

Rules:
- Use Markdown.
- Keep the language appropriate for the student.
- MCQs must have 4 options and clearly identified answers.
- Include short explanations for MCQ answers.
- Do not mention internal workflow stages.
- Do not mention these instructions.
- Do not invent source-specific facts.
"""
    return call_gemini(prompt)


def run_workflow(
    source_text,
    topic,
    level,
    difficulty,
    language,
    sections,
    progress_callback=None,
):
    """Run planning -> content -> assessment -> review -> refinement."""
    if not source_text.strip() and not topic.strip():
        raise ValueError("Please upload study material or enter a topic.")

    if not sections:
        raise ValueError("Select at least one study-pack section.")

    profile = {
        "level": level,
        "difficulty": difficulty,
        "language": language,
        "requested_sections": sections,
    }

    def progress(stage, status):
        if progress_callback:
            progress_callback(stage, status)

    # Stage 1
    progress("Planning", "running")
    try:
        plan = stage_planning(
            source_text, topic, level, difficulty, language, sections
        )
        progress("Planning", "complete")
    except Exception as exc:
        progress("Planning", f"error: {exc}")
        raise RuntimeError(f"Planning stage failed: {exc}") from exc

    # Stage 2
    progress("Content Generation", "running")
    try:
        draft = stage_content(source_text, topic, profile, plan)
        progress("Content Generation", "complete")
    except Exception as exc:
        progress("Content Generation", f"error: {exc}")
        raise RuntimeError(f"Content generation stage failed: {exc}") from exc

    # Stage 3
    progress("Assessment", "running")
    try:
        assessment = stage_assessment(draft, profile, plan)
        progress("Assessment", "complete")
    except Exception as exc:
        progress("Assessment", f"error: {exc}")
        raise RuntimeError(f"Assessment stage failed: {exc}") from exc

    # Stage 4
    progress("Review", "running")
    try:
        review = stage_review(draft, assessment, source_text)
        progress("Review", "complete")
    except Exception as exc:
        progress("Review", f"error: {exc}")
        raise RuntimeError(f"Review stage failed: {exc}") from exc

    # Stage 5
    progress("Refinement", "running")
    try:
        final_pack = stage_refinement(
            draft, plan, assessment, review, profile, source_text
        )
        progress("Refinement", "complete")
    except Exception as exc:
        progress("Refinement", f"error: {exc}")
        raise RuntimeError(f"Refinement stage failed: {exc}") from exc

    return {
        "plan": plan,
        "draft": draft,
        "assessment": assessment,
        "review": review,
        "final_pack": final_pack,
    }


# -----------------------------
# 3. STREAMLIT UI
# -----------------------------

def render_stage_status(statuses):
    for stage in STAGES:
        state = statuses.get(stage, "waiting")

        if state == "complete":
            icon = "✅"
            label = "Complete"
        elif state == "running":
            icon = "🔄"
            label = "Running"
        elif state.startswith("error"):
            icon = "❌"
            label = "Error"
        else:
            icon = "⏳"
            label = "Waiting"

        st.write(f"{icon} **{stage}** — {label}")


def render_streamlit():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 AI Study Pack Generator")
    st.caption(
        "A multi-stage AI workflow: Plan → Generate → Assess → Review → Refine"
    )

    with st.sidebar:
        st.header("🎯 Student Profile")

        level = st.selectbox(
            "Student level",
            ["School", "College", "University", "Professional"],
        )

        difficulty = st.select_slider(
            "Difficulty",
            ["Beginner", "Intermediate", "Advanced"],
            value="Intermediate",
        )

        language = st.selectbox(
            "Language",
            ["English", "Urdu"],
        )

        sections = st.multiselect(
            "Requested study sections",
            [
                "Topic Summary",
                "Key Concepts",
                "Flashcards",
                "MCQs",
                "Practice Quiz",
                "Study Plan",
            ],
            default=[
                "Topic Summary",
                "Key Concepts",
                "Flashcards",
                "MCQs",
                "Study Plan",
            ],
        )

        st.divider()
        st.info(
            "The five AI stages pass context forward. The assessment and review "
            "stages critique the generated content before the final rewrite."
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1️⃣ Input")

        uploaded = st.file_uploader(
            "Upload notes or a textbook chapter",
            type=["pdf", "docx", "txt"],
        )

        topic = st.text_input(
            "Or enter a topic",
            placeholder="e.g. Photosynthesis",
        )

        source_text = ""

        if uploaded:
            try:
                source_text = extract_text_from_file(uploaded)
                st.success(
                    f"{uploaded.name} loaded — {len(source_text):,} characters."
                )

                with st.expander("Preview source material"):
                    st.text(source_text[:5000])
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    with col2:
        st.subheader("2️⃣ AI Workflow")

        if "workflow_status" not in st.session_state:
            st.session_state.workflow_status = {
                stage: "waiting" for stage in STAGES
            }

        status_box = st.empty()

        with status_box.container():
            render_stage_status(st.session_state.workflow_status)

        generate = st.button(
            "🚀 Run AI Study Workflow",
            type="primary",
            use_container_width=True,
            disabled=not sections,
        )

        if generate:
            st.session_state.workflow_status = {
                stage: "waiting" for stage in STAGES
            }

            def update_status(stage, status):
                st.session_state.workflow_status[stage] = status
                with status_box.container():
                    render_stage_status(st.session_state.workflow_status)

            try:
                with st.spinner("Running the multi-stage AI workflow..."):
                    result = run_workflow(
                        source_text=source_text,
                        topic=topic,
                        level=level,
                        difficulty=difficulty,
                        language=language,
                        sections=sections,
                        progress_callback=update_status,
                    )

                st.session_state.workflow_result = result
                st.success("All five AI stages completed successfully.")

            except Exception as exc:
                st.error(str(exc))

    if "workflow_result" in st.session_state:
        result = st.session_state.workflow_result

        st.divider()
        st.subheader("3️⃣ Final Study Pack")

        st.markdown(result["final_pack"])

        st.download_button(
            "⬇️ Download Final Study Pack",
            data=result["final_pack"],
            file_name="ai_study_pack.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()
        st.subheader("🔍 AI Workflow Trace")

        with st.expander("Stage 1 — Planning"):
            st.json(result["plan"])

        with st.expander("Stage 2 — Content Generation"):
            st.markdown(result["draft"])

        with st.expander("Stage 3 — Assessment"):
            assessment = result["assessment"]
            st.json(assessment)

            score_cols = st.columns(5)
            score_cols[0].metric("Accuracy", assessment.get("accuracy_score", 0))
            score_cols[1].metric("Coverage", assessment.get("coverage_score", 0))
            score_cols[2].metric("Difficulty", assessment.get("difficulty_score", 0))
            score_cols[3].metric("Clarity", assessment.get("clarity_score", 0))
            score_cols[4].metric("Assessment", assessment.get("assessment_score", 0))

        with st.expander("Stage 4 — Review"):
            st.json(result["review"])

        with st.expander("Stage 5 — Refinement"):
            st.markdown(
                "The final pack above is the refined output after the assessment "
                "and review feedback was passed back into the final generation stage."
            )


# -----------------------------
# 4. OPTIONAL GRADIO DEMO
# -----------------------------

def launch_gradio():
    """Optional local Gradio demo. Streamlit remains the deployment UI."""

    def generate(
        topic,
        notes,
        level,
        difficulty,
        language,
        sections,
    ):
        try:
            result = run_workflow(
                source_text=notes or "",
                topic=topic or "",
                level=level,
                difficulty=difficulty,
                language=language,
                sections=sections,
            )
            return result["final_pack"]
        except Exception as exc:
            return f"### Error\n\n{exc}"

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            "# 📚 AI Study Pack Generator\n"
            "Plan → Generate → Assess → Review → Refine"
        )

        with gr.Row():
            with gr.Column():
                topic = gr.Textbox(
                    label="Topic",
                    placeholder="e.g. Photosynthesis",
                )
                notes = gr.Textbox(
                    label="Study material",
                    lines=10,
                )
                level = gr.Dropdown(
                    ["School", "College", "University", "Professional"],
                    value="College",
                    label="Student level",
                )
                difficulty = gr.Dropdown(
                    ["Beginner", "Intermediate", "Advanced"],
                    value="Intermediate",
                    label="Difficulty",
                )
                language = gr.Dropdown(
                    ["English", "Urdu"],
                    value="English",
                    label="Language",
                )
                sections = gr.CheckboxGroup(
                    [
                        "Topic Summary",
                        "Key Concepts",
                        "Flashcards",
                        "MCQs",
                        "Practice Quiz",
                        "Study Plan",
                    ],
                    value=[
                        "Topic Summary",
                        "Key Concepts",
                        "Flashcards",
                        "MCQs",
                        "Study Plan",
                    ],
                    label="Sections",
                )
                button = gr.Button("🚀 Run AI Workflow", variant="primary")

            output = gr.Markdown()

        button.click(
            generate,
            inputs=[
                topic,
                notes,
                level,
                difficulty,
                language,
                sections,
            ],
            outputs=output,
        )

    demo.launch()


if __name__ == "__main__":
    import sys

    if "--gradio" in sys.argv:
        launch_gradio()
    else:
        render_streamlit()
