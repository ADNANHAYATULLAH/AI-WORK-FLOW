import streamlit as st

from utils import extract_text_from_file
from workflow import WORKFLOW_STAGES, run_workflow


st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)


def show_workflow_status(statuses):
    """Display the status of each AI workflow stage."""
    for stage in WORKFLOW_STAGES:
        status = statuses.get(stage, "waiting")

        if status == "complete":
            icon, label = "✅", "Complete"
        elif status == "running":
            icon, label = "🔄", "Running"
        elif status.startswith("error"):
            icon, label = "❌", "Error"
        else:
            icon, label = "⏳", "Waiting"

        st.write(f"{icon} **{stage}** — {label}")


if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = {
        stage: "waiting" for stage in WORKFLOW_STAGES
    }


st.title("📚 AI Study Pack Generator")
st.subheader("Personalized learning through a 5-stage AI workflow")
st.caption("Plan → Generate → Assess → Review → Refine")


with st.sidebar:
    st.header("🎯 Student Profile")

    level = st.selectbox(
        "Student level",
        ["School", "College", "University", "Professional"],
        index=1,
    )

    difficulty = st.select_slider(
        "Difficulty",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Intermediate",
    )

    language = st.selectbox(
        "Language",
        ["English", "Urdu"],
    )

    sections = st.multiselect(
        "Study-pack sections",
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
        "The study material passes through five AI stages. "
        "Assessment and review feedback are passed into the final refinement stage."
    )


left, right = st.columns([1, 1])


with left:
    st.subheader("1️⃣ Study Material")

    uploaded_file = st.file_uploader(
        "Upload notes or a textbook chapter",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, and TXT.",
    )

    topic = st.text_input(
        "Or enter a topic",
        placeholder="Example: Photosynthesis",
    )

    source_text = ""

    if uploaded_file:
        try:
            source_text = extract_text_from_file(
                uploaded_file
            )

            st.success(
                f"Loaded {uploaded_file.name} "
                f"({len(source_text):,} characters)"
            )

            with st.expander("Preview source material"):
                st.text(source_text[:5000])

        except Exception as exc:
            st.error(
                f"File processing error: {exc}"
            )


with right:
    st.subheader("2️⃣ AI Workflow")

    status_placeholder = st.empty()

    with status_placeholder.container():
        show_workflow_status(
            st.session_state.workflow_status
        )

    generate_button = st.button(
        "🚀 Generate Study Pack",
        type="primary",
        use_container_width=True,
        disabled=not sections,
    )

    if generate_button:
        st.session_state.workflow_result = None
        st.session_state.workflow_status = {
            stage: "waiting"
            for stage in WORKFLOW_STAGES
        }

        def progress_callback(stage, status):
            st.session_state.workflow_status[
                stage
            ] = status

            with status_placeholder.container():
                show_workflow_status(
                    st.session_state.workflow_status
                )

        try:
            with st.spinner(
                "Running the five-stage AI workflow..."
            ):
                result = run_workflow(
                    source_text=source_text,
                    topic=topic,
                    level=level,
                    difficulty=difficulty,
                    language=language,
                    sections=sections,
                    progress_callback=progress_callback,
                )

            st.session_state.workflow_result = result

            st.success(
                "🎉 All five workflow stages completed successfully!"
            )

        except Exception as exc:
            st.error(str(exc))


result = st.session_state.workflow_result


if result:
    st.divider()

    st.subheader(
        "3️⃣ Final Personalized Study Pack"
    )

    st.markdown(
        result["final_pack"]
    )

    st.download_button(
        label="⬇️ Download Study Pack",
        data=result["final_pack"],
        file_name="ai_study_pack.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()

    st.subheader("🔍 AI Workflow Trace")

    st.caption(
        "The trace demonstrates context passing between stages "
        "and is useful for your hackathon presentation."
    )

    with st.expander("① Planning Output"):
        st.json(result["plan"])

    with st.expander("② Content Generation Output"):
        st.markdown(result["draft"])

    with st.expander("③ Assessment Output"):
        assessment = result["assessment"]

        score_columns = st.columns(6)

        score_keys = [
            ("Accuracy", "accuracy_score"),
            ("Coverage", "coverage_score"),
            ("Difficulty", "difficulty_score"),
            ("Clarity", "clarity_score"),
            ("Assessment", "assessment_score"),
            ("Overall", "overall_score"),
        ]

        for column, (label, key) in zip(
            score_columns, score_keys
        ):
            column.metric(
                label,
                assessment.get(key, 0),
            )

        st.json(assessment)

    with st.expander("④ Review Output"):
        st.json(result["review"])

    with st.expander("⑤ Refinement"):
        st.success(
            "The final study pack was generated after applying "
            "assessment and review feedback."
        )
