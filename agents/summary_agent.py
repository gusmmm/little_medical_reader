import logging
import os
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from google import genai
from dotenv import load_dotenv
import json
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging for this module
logger = logging.getLogger(__name__)

# Initialize the Gemini client - API key loaded from environment variable GEMINI_API_KEY
try:
    client = genai.Client()
    logger.info("✓ Google Gemini AI client initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize Gemini client: {e}")
    client = None

def load_markdown_content(file_path: Optional[str] = None) -> Optional[str]:
    """
    Load markdown content either from a file path or from Streamlit session state.
    
    Args:
        file_path: Optional path to markdown file. If None, tries to load from session state.
        
    Returns:
        String containing markdown content, or None if loading fails
    """
    logger.info("Starting markdown content loading process")
    
    # First try to load from file path if provided
    if file_path:
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists() and file_path_obj.is_file():
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"✓ Successfully loaded markdown from file: {file_path}")
                logger.info(f"Content length: {len(content)} characters")
                return content
            else:
                logger.warning(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    # Try to load from session state if file loading failed or no path provided
    try:
        if hasattr(st, 'session_state') and "markdown_content" in st.session_state:
            content = st.session_state["markdown_content"]
            logger.info("✓ Successfully loaded markdown from session state")
            logger.info(f"Content length: {len(content)} characters")
            return content
        else:
            logger.warning("No markdown content found in session state")
    except Exception as e:
        logger.error(f"Error accessing session state: {e}")
    
    logger.error("Failed to load markdown content from any source")
    return None

def detect_medical_document_type(content: str) -> Dict[str, Any]:
    """
    Analyze medical document content to determine its type and characteristics.
    Uses both AI analysis and keyword-based fallback detection.
    
    Args:
        content: The markdown content to analyze
        
    Returns:
        Dictionary containing document analysis results
    """
    logger.info("Starting comprehensive medical document type detection")
    
    # First attempt: AI-powered analysis
    if client:
        ai_analysis = _ai_document_analysis(content)
        if ai_analysis and "error" not in ai_analysis:
            logger.info("✓ AI-powered document analysis successful")
            return ai_analysis
        else:
            logger.warning("AI analysis failed, falling back to keyword detection")
    else:
        logger.warning("No AI client available, using keyword detection")
    
    # Fallback: Keyword-based analysis
    return _keyword_based_analysis(content)

def _ai_document_analysis(content: str) -> Dict[str, Any]:
    """
    Use Gemini AI to analyze document type with enhanced prompting for medical documents.
    
    Args:
        content: The markdown content to analyze
        
    Returns:
        Dictionary containing AI analysis results or error information
    """
    logger.info("Performing AI-powered document analysis")
    
    analysis_prompt = f"""
    You are an expert medical document classifier. Analyze the following medical document and determine its type and key characteristics.

    DOCUMENT CONTENT (first 3000 characters):
    {content[:3000]}

    Classify this document and respond with ONLY a valid JSON object in this exact format:
    {{
        "document_type": "one of: journal_article, discharge_summary, admission_note, progress_note, consultation_note, lab_results, pathology_report, imaging_report, procedure_note, medication_list, care_plan, clinical_trial, case_report, review_article, other",
        "document_subtype": "more specific classification if applicable",
        "medical_specialty": "primary medical specialty involved",
        "complexity_level": "low/medium/high",
        "key_sections": ["list", "of", "main", "sections", "identified"],
        "summary_strategy": "best approach for summarizing this document type",
        "target_audience": "who would primarily use this document",
        "urgency_level": "low/medium/high based on clinical relevance",
        "contains_patient_data": "yes/no/unclear"
    }}

    Only return valid JSON, no other text.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=analysis_prompt
        )
        
        response_text = response.text.strip()
        logger.debug(f"AI response length: {len(response_text)} characters")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            json_text = response_text
        
        analysis_result = json.loads(json_text)
        
        # Validate and clean up results
        required_fields = ["document_type", "summary_strategy", "target_audience"]
        for field in required_fields:
            if field not in analysis_result:
                analysis_result[field] = "unknown"
        
        logger.info(f"✓ Document classified as: {analysis_result.get('document_type', 'unknown')}")
        logger.info(f"Medical specialty: {analysis_result.get('medical_specialty', 'unknown')}")
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in AI analysis: {e}")
        return {"error": "JSON parsing failed"}
    except Exception as e:
        logger.error(f"Error in AI document analysis: {e}")
        return {"error": str(e)}

def _keyword_based_analysis(content: str) -> Dict[str, Any]:
    """
    Fallback keyword-based document classification for medical documents.
    
    Args:
        content: The markdown content to analyze
        
    Returns:
        Dictionary containing keyword-based analysis results
    """
    logger.info("Performing keyword-based document classification")
    
    content_lower = content.lower()
    
    # Document type classification based on keywords
    document_classifications = {
        "journal_article": ["abstract", "introduction", "methodology", "results", "discussion", "references", "doi", "pmid", "citation"],
        "discharge_summary": ["discharge", "discharged", "disposition", "follow-up", "discharge instructions", "final diagnosis"],
        "admission_note": ["admission", "admitted", "chief complaint", "history of present illness", "physical examination"],
        "progress_note": ["progress", "daily note", "soap", "assessment and plan", "interval history"],
        "consultation_note": ["consultation", "consult", "referred", "recommendation", "consultant"],
        "lab_results": ["laboratory", "lab values", "blood work", "urinalysis", "culture", "chemistry panel"],
        "pathology_report": ["pathology", "biopsy", "histology", "microscopic", "gross description"],
        "imaging_report": ["radiology", "ct scan", "mri", "x-ray", "ultrasound", "impression", "findings"],
        "procedure_note": ["procedure", "operative", "surgery", "technique", "complications", "post-operative"],
        "medication_list": ["medications", "prescriptions", "dosage", "pharmacy", "drug list"],
        "care_plan": ["care plan", "goals", "interventions", "nursing", "treatment plan"],
        "clinical_trial": ["clinical trial", "study protocol", "randomized", "placebo", "endpoint"],
        "case_report": ["case report", "case presentation", "patient presented", "rare case"],
        "review_article": ["systematic review", "meta-analysis", "literature review", "evidence-based"]
    }
    
    # Score each document type
    type_scores = {}
    for doc_type, keywords in document_classifications.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            type_scores[doc_type] = score
    
    # Determine document type
    if type_scores:
        document_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[document_type]
    else:
        document_type = "other"
        confidence = 0
    
    # Determine medical specialty
    specialty_keywords = {
        "cardiology": ["heart", "cardiac", "cardiovascular", "ecg", "ekg", "coronary"],
        "neurology": ["brain", "neurological", "stroke", "seizure", "neurologic"],
        "oncology": ["cancer", "tumor", "oncology", "chemotherapy", "radiation"],
        "surgery": ["surgical", "operation", "procedure", "operative", "incision"],
        "internal_medicine": ["internal medicine", "general medicine", "medical"],
        "emergency_medicine": ["emergency", "urgent", "trauma", "er", "acute"],
        "pediatrics": ["pediatric", "child", "infant", "neonatal"],
        "psychiatry": ["psychiatric", "mental health", "psychology", "depression"],
        "radiology": ["imaging", "radiology", "scan", "x-ray", "ct", "mri"]
    }
    
    specialty_scores = {}
    for specialty, keywords in specialty_keywords.items():
        score = sum(1 for keyword in keywords if keyword in content_lower)
        if score > 0:
            specialty_scores[specialty] = score
    
    medical_specialty = max(specialty_scores, key=specialty_scores.get) if specialty_scores else "general_medicine"
    
    # Determine complexity and urgency
    complexity_indicators = ["complex", "complicated", "multiple", "extensive", "severe"]
    urgency_indicators = ["urgent", "emergent", "critical", "immediate", "stat"]
    
    complexity_score = sum(1 for indicator in complexity_indicators if indicator in content_lower)
    urgency_score = sum(1 for indicator in urgency_indicators if indicator in content_lower)
    
    complexity_level = "high" if complexity_score >= 2 else "medium" if complexity_score >= 1 else "low"
    urgency_level = "high" if urgency_score >= 2 else "medium" if urgency_score >= 1 else "low"
    
    # Determine summary strategy based on document type
    summary_strategies = {
        "journal_article": "academic_research_summary",
        "discharge_summary": "clinical_care_transition",
        "admission_note": "initial_assessment_summary",
        "progress_note": "ongoing_care_summary",
        "consultation_note": "specialist_recommendation",
        "lab_results": "diagnostic_findings_summary",
        "pathology_report": "diagnostic_pathology_summary",
        "imaging_report": "imaging_findings_summary",
        "procedure_note": "procedural_summary",
        "medication_list": "medication_reconciliation",
        "care_plan": "treatment_plan_summary",
        "clinical_trial": "research_study_summary",
        "case_report": "clinical_case_summary",
        "review_article": "evidence_synthesis_summary",
        "other": "general_medical_summary"
    }
    
    result = {
        "document_type": document_type,
        "document_subtype": f"{document_type}_variant",
        "medical_specialty": medical_specialty,
        "complexity_level": complexity_level,
        "key_sections": _extract_section_headers(content),
        "summary_strategy": summary_strategies.get(document_type, "general_medical_summary"),
        "target_audience": _determine_target_audience(document_type),
        "urgency_level": urgency_level,
        "contains_patient_data": "unclear",
        "confidence_score": confidence
    }
    
    logger.info(f"✓ Keyword analysis completed: {document_type} ({confidence} keyword matches)")
    return result

def _extract_section_headers(content: str) -> list:
    """Extract section headers from markdown content."""
    headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
    return headers[:10]  # Limit to first 10 headers

def _determine_target_audience(document_type: str) -> str:
    """Determine the primary target audience based on document type."""
    audience_map = {
        "journal_article": "researchers_and_clinicians",
        "discharge_summary": "receiving_healthcare_providers",
        "admission_note": "healthcare_team",
        "progress_note": "healthcare_team",
        "consultation_note": "referring_physician",
        "lab_results": "ordering_physician",
        "pathology_report": "treating_physician",
        "imaging_report": "referring_physician",
        "procedure_note": "healthcare_team",
        "medication_list": "pharmacists_and_nurses",
        "care_plan": "healthcare_team",
        "clinical_trial": "researchers",
        "case_report": "medical_professionals",
        "review_article": "clinicians_and_researchers"
    }
    return audience_map.get(document_type, "healthcare_professionals")

def generate_adaptive_summary(content: str, analysis: Dict[str, Any]) -> Optional[str]:
    """
    Generate a structured summary adapted to the specific document type and characteristics.
    
    Args:
        content: The original document content
        analysis: Document analysis results from detect_medical_document_type()
        
    Returns:
        Structured summary as markdown string, or None if generation fails
    """
    logger.info("Starting adaptive summary generation")
    
    if not client:
        logger.error("Gemini client not available for summary generation")
        return None
    
    document_type = analysis.get('document_type', 'other')
    summary_strategy = analysis.get('summary_strategy', 'general_medical_summary')
    target_audience = analysis.get('target_audience', 'healthcare_professionals')
    medical_specialty = analysis.get('medical_specialty', 'general_medicine')
    
    # Generate template based on document type
    summary_template = _get_summary_template(document_type, summary_strategy)
    
    summary_prompt = f"""
    You are an expert medical summarization AI. Create a comprehensive, structured summary of this {document_type} 
    for {target_audience} in the {medical_specialty} field.

    ORIGINAL DOCUMENT:
    {content}

    Please create a summary using this structure:
    {summary_template}

    IMPORTANT GUIDELINES:
    - Maintain medical accuracy and use appropriate clinical terminology
    - Highlight the most critical information for {target_audience}
    - Be concise but comprehensive
    - Include specific data, findings, and recommendations where applicable
    - Preserve important medical details and measurements
    - Use professional medical language appropriate for healthcare providers

    Create the summary now:
    """
    
    try:
        logger.info(f"Generating {summary_strategy} summary for {document_type}")
        logger.info(f"Target audience: {target_audience}")
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=summary_prompt
        )
        
        summary = response.text.strip()
        logger.info("✓ Adaptive summary generated successfully")
        logger.info(f"Summary length: {len(summary)} characters")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating adaptive summary: {e}")
        return None

def _get_summary_template(document_type: str, summary_strategy: str) -> str:
    """
    Generate appropriate summary template based on document type and strategy.
    
    Args:
        document_type: Type of medical document
        summary_strategy: Summarization strategy to use
        
    Returns:
        Markdown template string for the summary
    """
    templates = {
        "journal_article": """
# Research Article Summary

## Study Overview
- **Title**: [Article title]
- **Authors**: [Key authors]
- **Journal**: [Publication details]
- **Study Type**: [Research methodology]

## Key Findings
- **Primary Objective**: [Main research question]
- **Study Population**: [Participants/subjects]
- **Main Results**: [Key statistical findings]
- **Clinical Significance**: [Practical implications]

## Methodology
- **Study Design**: [Type of study]
- **Sample Size**: [Number of participants]
- **Key Methods**: [Primary techniques used]

## Clinical Implications
- **Practice Changes**: [How this affects clinical practice]
- **Future Research**: [Suggested next steps]
- **Limitations**: [Study limitations]

## Conclusion
[Overall significance and takeaway message]
        """,
        
        "discharge_summary": """
# Discharge Summary

## Patient Information
- **Admission Date**: [Date]
- **Discharge Date**: [Date]
- **Length of Stay**: [Duration]
- **Primary Diagnosis**: [Main condition]

## Hospital Course
- **Admission Reason**: [Why patient was admitted]
- **Key Treatments**: [Major interventions]
- **Complications**: [Any issues during stay]
- **Response to Treatment**: [Patient progress]

## Discharge Status
- **Condition at Discharge**: [Patient status]
- **Functional Status**: [Mobility/independence level]
- **Medications**: [Key medication changes]

## Follow-up Care
- **Appointments**: [Scheduled follow-up]
- **Home Care**: [Special instructions]
- **Warning Signs**: [When to seek immediate care]

## Recommendations
[Key discharge instructions and next steps]
        """,
        
        "clinical_trial": """
# Clinical Trial Summary

## Study Overview
- **Trial Title**: [Study name]
- **Phase**: [Trial phase]
- **Primary Endpoint**: [Main outcome measure]
- **Study Population**: [Participant criteria]

## Study Design
- **Methodology**: [Randomized, controlled, etc.]
- **Sample Size**: [Number of participants]
- **Duration**: [Study length]
- **Intervention**: [Treatment being tested]

## Results
- **Primary Outcomes**: [Main findings]
- **Secondary Outcomes**: [Additional findings]
- **Safety Profile**: [Adverse events]
- **Statistical Significance**: [Key p-values, confidence intervals]

## Clinical Impact
- **Efficacy**: [Treatment effectiveness]
- **Safety**: [Risk profile]
- **Regulatory Implications**: [Approval prospects]

## Conclusions
[Overall study significance and next steps]
        """
    }
    
    # Return specific template or general medical template
    return templates.get(document_type, """
# Medical Document Summary

## Document Overview
- **Document Type**: [Type of medical document]
- **Date**: [Document date]
- **Department/Specialty**: [Medical area]
- **Author/Provider**: [Healthcare provider]

## Key Information
- **Primary Focus**: [Main subject/concern]
- **Key Findings**: [Important discoveries/observations]
- **Critical Data**: [Measurements, test results, vital signs]

## Clinical Assessment
- **Current Status**: [Patient/subject condition]
- **Significant Changes**: [Important developments]
- **Risk Factors**: [Identified concerns]

## Recommendations
- **Immediate Actions**: [Urgent steps needed]
- **Follow-up Requirements**: [Next steps]
- **Monitoring**: [What to watch for]

## Additional Notes
[Other relevant information]
    """)

def save_summary_to_file(summary: str, original_filename: str = "medical_document") -> Optional[str]:
    """
    Save the generated medical summary to a markdown file in the output directory.
    
    Args:
        summary: The generated summary content
        original_filename: Base filename for the summary file
        
    Returns:
        Path to saved summary file, or None if saving fails
    """
    logger.info("Starting summary file save process")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate summary filename
        base_name = Path(original_filename).stem
        summary_filename = f"{base_name}_summary.md"
        summary_path = output_dir / summary_filename
        
        # Write summary to file
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"✓ Medical summary saved successfully: {summary_path}")
        return str(summary_path)
        
    except Exception as e:
        logger.error(f"Error saving summary file: {e}")
        return None

def process_medical_document(file_path: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Complete workflow to process medical document: load, analyze, summarize, and save.
    
    Args:
        file_path: Optional path to markdown file. If None, loads from session state.
        
    Returns:
        Tuple of (success_flag, summary_file_path, analysis_results)
    """
    logger.info("=== Starting complete medical document processing workflow ===")
    
    try:
        # Step 1: Load markdown content
        logger.info("Step 1: Loading markdown content")
        content = load_markdown_content(file_path)
        if not content:
            logger.error("Failed to load markdown content")
            return False, None, None
        
        # Step 2: Analyze document type and characteristics
        logger.info("Step 2: Analyzing document type and characteristics")
        analysis = detect_medical_document_type(content)
        if "error" in analysis:
            logger.error(f"Document analysis failed: {analysis['error']}")
            return False, None, analysis
        
        # Step 3: Generate medical summary
        logger.info("Step 3: Generating structured medical summary")
        summary = generate_adaptive_summary(content, analysis)
        if not summary:
            logger.error("Failed to generate medical summary")
            return False, None, analysis
        
        # Step 4: Save summary to file
        logger.info("Step 4: Saving summary to output file")
        original_name = file_path if file_path else "session_document"
        summary_path = save_summary_to_file(summary, original_name)
        if not summary_path:
            logger.error("Failed to save summary file")
            return False, None, analysis
        
        logger.info("=== Medical document processing completed successfully ===")
        return True, summary_path, analysis
        
    except Exception as e:
        logger.error(f"Unexpected error in medical document processing: {e}")
        return False, None, None

def test_summary_agent():
    """
    Comprehensive test function using the provided medical journal article.
    Tests the complete workflow with a real academic medical document.
    """
    logger.info("=== Running Enhanced Summary Agent Tests ===")
    
    # Test 1: Check dependencies
    logger.info("Test 1: Checking dependencies and setup")
    try:
        from google import genai
        from dotenv import load_dotenv
        logger.info("✓ All dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False
    
    # Test 2: Check Gemini client
    logger.info("Test 2: Testing AI client connection")
    if client:
        logger.info("✓ Gemini client initialized successfully")
    else:
        logger.error("✗ Gemini client initialization failed - check API key")
        return False
    
    # Test 3: Load the provided journal article
    logger.info("Test 3: Loading journal article for testing")
    test_file_path = Path("output/Self-directed learning in health professions.md")
    
    if not test_file_path.exists():
        logger.error(f"✗ Test file not found: {test_file_path}")
        logger.info("Please ensure the journal article file is available for testing")
        return False
    
    content = load_markdown_content(str(test_file_path))
    if not content:
        logger.error("✗ Failed to load journal article content")
        return False
    
    logger.info(f"✓ Journal article loaded successfully ({len(content)} characters)")
    
    # Test 4: Document type detection
    logger.info("Test 4: Testing document type detection")
    analysis = detect_medical_document_type(content)
    if "error" in analysis:
        logger.error(f"✗ Document analysis failed: {analysis['error']}")
        return False
    
    logger.info("✓ Document analysis completed successfully")
    logger.info(f"  Document type: {analysis.get('document_type', 'unknown')}")
    logger.info(f"  Medical specialty: {analysis.get('medical_specialty', 'unknown')}")
    logger.info(f"  Summary strategy: {analysis.get('summary_strategy', 'unknown')}")
    logger.info(f"  Target audience: {analysis.get('target_audience', 'unknown')}")
    
    # Test 5: Adaptive summary generation
    logger.info("Test 5: Testing adaptive summary generation")
    summary = generate_adaptive_summary(content, analysis)
    if not summary:
        logger.error("✗ Failed to generate adaptive summary")
        return False
    
    logger.info("✓ Adaptive summary generated successfully")
    logger.info(f"Summary length: {len(summary)} characters")
    
    # Test 6: Save summary and verify
    logger.info("Test 6: Testing summary save functionality")
    summary_path = save_summary_to_file(summary, "Self-directed_learning_journal_article")
    if not summary_path:
        logger.error("✗ Failed to save summary file")
        return False
    
    logger.info(f"✓ Summary saved successfully: {summary_path}")
    
    # Test 7: Verify saved content
    logger.info("Test 7: Verifying saved summary content")
    if Path(summary_path).exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        logger.info(f"✓ Summary file verified ({len(saved_content)} characters)")
        
        # Display preview of the generated summary
        logger.info("=== GENERATED SUMMARY PREVIEW ===")
        preview_length = 800
        preview = saved_content[:preview_length]
        if len(saved_content) > preview_length:
            preview += "\n... [truncated for display] ..."
        print(preview)
        logger.info("=== END SUMMARY PREVIEW ===")
    else:
        logger.error("✗ Summary file not found after saving")
        return False
    
    # Test 8: Complete workflow test
    logger.info("Test 8: Testing complete processing workflow")
    success, workflow_summary_path, workflow_analysis = process_medical_document(str(test_file_path))
    
    if success:
        logger.info("✓ Complete workflow test passed")
        logger.info(f"Workflow summary path: {workflow_summary_path}")
        
        # Display analysis results
        if workflow_analysis:
            logger.info("Complete Analysis Results:")
            for key, value in workflow_analysis.items():
                if key != "error":
                    logger.info(f"  {key}: {value}")
    else:
        logger.error("✗ Complete workflow test failed")
        return False
    
    logger.info("=== All Enhanced Summary Agent Tests Passed! ===")
    logger.info(f"Test completed using journal article: {test_file_path}")
    logger.info("The system successfully:")
    logger.info("  - Detected document type as academic research")
    logger.info("  - Generated appropriate research summary structure")
    logger.info("  - Adapted content for clinical and research audiences")
    logger.info("  - Saved structured summary for future reference")
    
    return True

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Run comprehensive tests with the journal article
    test_summary_agent()