import json
from utils import call_gemini
WORKFLOW_STAGES=['Planning','Content Generation','Assessment','Review','Refinement']
def stage_planning(source,topic,profile,sections):
    p=f'''You are Stage 1, an expert instructional planner. Create a personalized study-pack PLAN, not the final pack.\nProfile: {json.dumps(profile)}\nSections: {sections}\nTopic: {topic or "infer from source"}\nSource: {source[:24000]}\nReturn ONLY JSON with title, topic, learning_objectives, key_concepts, content_sections, question_count, study_plan_days, difficulty_guidance. Base source-specific planning on supplied material.'''
    return call_gemini(p,True)
def stage_content(source,topic,profile,plan,sections):
    p=f'''You are Stage 2, an expert teacher. Create the first draft using the plan. Profile: {json.dumps(profile)}\nSections: {sections}\nPlan: {json.dumps(plan)}\nTopic: {topic or plan.get("topic","Unknown")}\nSource: {source[:24000]}\nCreate requested summary, key concepts, flashcards, MCQs, quiz and study plan as applicable. Use Markdown. MCQs need four options, answers and explanations. Do not invent source-specific facts.'''
    return call_gemini(p)
def stage_assessment(draft,plan,profile,source):
    p=f'''You are Stage 3, an educational quality evaluator. Evaluate draft against plan, profile and source. Profile: {json.dumps(profile)}\nPlan: {json.dumps(plan)}\nSource: {source[:18000]}\nDraft: {draft[:30000]}\nReturn ONLY JSON with integer 0-100 scores: accuracy_score, coverage_score, difficulty_score, clarity_score, assessment_score, overall_score, plus strengths, issues, missing_topics, question_issues, recommended_fixes.'''
    return call_gemini(p,True)
def stage_review(draft,assessment,plan,source):
    p=f'''You are Stage 4, senior reviewer. Plan: {json.dumps(plan)}\nAssessment: {json.dumps(assessment)}\nSource: {source[:18000]}\nDraft: {draft[:28000]}\nReturn ONLY JSON with pass, priority_fixes, source_conflicts, format_fixes, final_instructions. Prioritize material improvements.'''
    return call_gemini(p,True)
def stage_refinement(draft,plan,assessment,review,profile,source,sections):
    p=f'''You are Stage 5, final educational editor. Profile: {json.dumps(profile)}\nSections: {sections}\nPlan: {json.dumps(plan)}\nAssessment: {json.dumps(assessment)}\nReview: {json.dumps(review)}\nSource: {source[:22000]}\nDraft: {draft[:30000]}\nApply feedback and produce final polished study pack. Use Markdown headings for Learning Objectives, Topic Summary, Key Concepts, Flashcards, Multiple-Choice Questions, Practice Quiz, Study Plan, Quick Revision Checklist; include only requested sections. Do not mention internal workflow. Do not invent source-specific facts.'''
    return call_gemini(p)
def run_workflow(source_text,topic,level,difficulty,language,sections,progress_callback=None):
    if not source_text.strip() and not topic.strip(): raise ValueError('Please enter a topic or upload study material.')
    if not sections: raise ValueError('Select at least one study-pack section.')
    profile={'level':level,'difficulty':difficulty,'language':language}
    def u(s,x):
        if progress_callback: progress_callback(s,x)
    u('Planning','running')
    try: plan=stage_planning(source_text,topic,profile,sections); u('Planning','complete')
    except Exception as e: u('Planning',f'error: {e}'); raise RuntimeError(f'Planning stage failed: {e}') from e
    u('Content Generation','running')
    try: draft=stage_content(source_text,topic,profile,plan,sections); u('Content Generation','complete')
    except Exception as e: u('Content Generation',f'error: {e}'); raise RuntimeError(f'Content Generation stage failed: {e}') from e
    u('Assessment','running')
    try: assessment=stage_assessment(draft,plan,profile,source_text); u('Assessment','complete')
    except Exception as e: u('Assessment',f'error: {e}'); raise RuntimeError(f'Assessment stage failed: {e}') from e
    u('Review','running')
    try: review=stage_review(draft,assessment,plan,source_text); u('Review','complete')
    except Exception as e: u('Review',f'error: {e}'); raise RuntimeError(f'Review stage failed: {e}') from e
    u('Refinement','running')
    try: final=stage_refinement(draft,plan,assessment,review,profile,source_text,sections); u('Refinement','complete')
    except Exception as e: u('Refinement',f'error: {e}'); raise RuntimeError(f'Refinement stage failed: {e}') from e
    return {'plan':plan,'draft':draft,'assessment':assessment,'review':review,'final_pack':final}
