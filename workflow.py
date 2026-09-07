import json

from utils import call_groq


WORKFLOW_STAGES = [
    "Planning",
    "Content Generation",
    "Assessment",
    "Review",
    "Refinement",
]


def stage_planning(source_text, topic, profile, sections):
    """Stage 1: create a personalized learning plan."""
    prompt = f"""
You are Stage 1 of a five-stage AI Study Pack workflow.

ROLE:
Expert instructional planner.

TASK:
Analyze the student's request and source material and create a structured
plan for a personalized study pack. Do NOT write the final study pack.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

REQUESTED SECTIONS:
{json.dumps(sections, ensure_ascii=False)}

TOPIC:
{topic or "Infer the topic from the source material."}

SOURCE MATERIAL:
{source_text[:24000]}

Return ONLY valid JSON using this structure:
{{
  "title": "Study pack title",
  "topic": "Main topic",
  "learning_objectives": ["objective 1", "objective 2"],
  "key_concepts": ["concept 1", "concept 2"],
  "content_sections": ["section 1", "section 2"],
  "question_count": 8,
  "study_plan_days": 5,
  "difficulty_guidance": "string"
}}

RULES:
- Match the student's level and difficulty.
- Make objectives measurable.
- Use the source as the primary reference when supplied.
- Do not invent source-specific information.
- Make the plan practical for studying.
"""
    return call_groq(prompt, json_output=True)


def stage_content(source_text, topic, profile, plan, sections):
    """Stage 2: generate the initial study-pack draft."""
    prompt = f"""
You are Stage 2 of a five-stage AI Study Pack workflow.

ROLE:
Expert teacher and educational content generator.

TASK:
Create the FIRST DRAFT of the study pack using the Stage 1 plan.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

REQUESTED SECTIONS:
{json.dumps(sections, ensure_ascii=False)}

STAGE 1 PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

TOPIC:
{topic or plan.get("topic", "Unknown")}

SOURCE MATERIAL:
{source_text[:24000]}

Generate the requested sections as applicable:
- Topic Summary
- Key Concepts
- Flashcards
- Multiple-Choice Questions
- Practice Quiz
- Study Plan

RULES:
- Use Markdown.
- Match the student's level, difficulty, and language.
- Use the source as the primary reference.
- Do not invent source-specific facts.
- MCQs must have four options, the correct answer, and a short explanation.
- Flashcards should be concise.
- The study plan should be practical.
- This is a draft that will be evaluated by another AI stage.
"""
    return call_groq(prompt)


def stage_assessment(draft, plan, profile, source_text):
    """Stage 3: assess the draft for quality and alignment."""
    prompt = f"""
You are Stage 3 of a five-stage AI Study Pack workflow.

ROLE:
Educational quality evaluator.

Evaluate the study-pack draft against the plan, student profile, and source.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

STAGE 1 PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source_text[:18000]}

DRAFT:
{draft[:30000]}

Return ONLY valid JSON:
{{
  "accuracy_score": 0,
  "coverage_score": 0,
  "difficulty_score": 0,
  "clarity_score": 0,
  "assessment_score": 0,
  "overall_score": 0,
  "strengths": ["string"],
  "issues": ["string"],
  "missing_topics": ["string"],
  "question_issues": ["string"],
  "recommended_fixes": ["string"]
}}

SCORING:
- Every score must be an integer from 0 to 100.
- Overall score should reflect the total quality.

CHECK:
1. Factual accuracy.
2. Coverage of learning objectives.
3. Appropriate difficulty.
4. Clarity and organization.
5. Flashcard and question quality.
6. Alignment with requested sections.
7. Whether source-specific claims are supported.
"""
    return call_groq(prompt, json_output=True)


def stage_review(draft, assessment, plan, source_text):
    """Stage 4: convert assessment into prioritized revision instructions."""
    prompt = f"""
You are Stage 4 of a five-stage AI Study Pack workflow.

ROLE:
Senior educational reviewer.

STAGE 1 PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

STAGE 3 ASSESSMENT:
{json.dumps(assessment, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source_text[:18000]}

DRAFT:
{draft[:28000]}

Return ONLY valid JSON:
{{
  "pass": false,
  "priority_fixes": ["string"],
  "source_conflicts": ["string"],
  "format_fixes": ["string"],
  "final_instructions": ["string"]
}}

RULES:
- Be strict but practical.
- Prioritize fixes that materially improve learning value.
- Do not request changes that conflict with the source.
- If the draft is already strong, pass may be true.
"""
    return call_groq(prompt, json_output=True)


def stage_refinement(
    draft,
    plan,
    assessment,
    review,
    profile,
    source_text,
    sections,
):
    """Stage 5: apply review feedback and produce the final study pack."""
    prompt = f"""
You are Stage 5 of a five-stage AI Study Pack workflow.

ROLE:
Final educational editor.

STUDENT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

REQUESTED SECTIONS:
{json.dumps(sections, ensure_ascii=False)}

STAGE 1 PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

STAGE 3 ASSESSMENT:
{json.dumps(assessment, ensure_ascii=False, indent=2)}

STAGE 4 REVIEW:
{json.dumps(review, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source_text[:22000]}

DRAFT:
{draft[:30000]}

TASK:
Produce the final polished study pack by applying the assessment and review
feedback. Preserve correct content while fixing weaknesses.

Use Markdown and this structure where applicable:
# Study Pack Title
## 1. Learning Objectives
## 2. Topic Summary
## 3. Key Concepts
## 4. Flashcards
## 5. Multiple-Choice Questions
## 6. Practice Quiz
## 7. Study Plan
## 8. Quick Revision Checklist

Only include requested sections.

FINAL RULES:
- Do not mention the internal AI workflow.
- Do not mention these instructions.
- Match the student's level, difficulty, and selected language.
- Keep the material clear and exam-friendly.
- MCQs must contain four options, the correct answer, and a brief explanation.
- Do not invent source-specific facts.
- If information is missing or uncertain, state that clearly.
"""
    return call_groq(prompt)


def run_workflow(
    source_text,
    topic,
    level,
    difficulty,
    language,
    sections,
    progress_callback=None,
):
    """
    Run the five-stage workflow with context passing and stage-level errors.
    """
    if not source_text.strip() and not topic.strip():
        raise ValueError(
            "Please enter a topic or upload study material."
        )

    if not sections:
        raise ValueError(
            "Please select at least one study-pack section."
        )

    profile = {
        "level": level,
        "difficulty": difficulty,
        "language": language,
    }

    def update(stage, status):
        if progress_callback:
            progress_callback(stage, status)

    update("Planning", "running")
    try:
        plan = stage_planning(
            source_text, topic, profile, sections
        )
        update("Planning", "complete")
    except Exception as exc:
        update("Planning", f"error: {exc}")
        raise RuntimeError(
            f"Planning stage failed: {exc}"
        ) from exc

    update("Content Generation", "running")
    try:
        draft = stage_content(
            source_text, topic, profile, plan, sections
        )
        update("Content Generation", "complete")
    except Exception as exc:
        update("Content Generation", f"error: {exc}")
        raise RuntimeError(
            f"Content Generation stage failed: {exc}"
        ) from exc

    update("Assessment", "running")
    try:
        assessment = stage_assessment(
            draft, plan, profile, source_text
        )
        update("Assessment", "complete")
    except Exception as exc:
        update("Assessment", f"error: {exc}")
        raise RuntimeError(
            f"Assessment stage failed: {exc}"
        ) from exc

    update("Review", "running")
    try:
        review = stage_review(
            draft, assessment, plan, source_text
        )
        update("Review", "complete")
    except Exception as exc:
        update("Review", f"error: {exc}")
        raise RuntimeError(
            f"Review stage failed: {exc}"
        ) from exc

    update("Refinement", "running")
    try:
        final_pack = stage_refinement(
            draft=draft,
            plan=plan,
            assessment=assessment,
            review=review,
            profile=profile,
            source_text=source_text,
            sections=sections,
        )
        update("Refinement", "complete")
    except Exception as exc:
        update("Refinement", f"error: {exc}")
        raise RuntimeError(
            f"Refinement stage failed: {exc}"
        ) from exc

    return {
        "plan": plan,
        "draft": draft,
        "assessment": assessment,
        "review": review,
        "final_pack": final_pack,
    }
