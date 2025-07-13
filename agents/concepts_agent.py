from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logging
import streamlit as st
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging for this module
logger = logging.getLogger(__name__)

# Initialize the Gemini client - API key loaded from environment variable GEMINI_API_KEY
try:
    client = genai.Client()
    logger.info("✓ Google Gemini AI client initialized successfully for concepts extraction")
except Exception as e:
    logger.error(f"✗ Failed to initialize Gemini client: {e}")
    client = None


class MedicalConcept(BaseModel):
    """
    Medical concept model specifically designed for clinical knowledge extraction and Neo4j semantic graph generation.
    
    This model focuses on four specific types of medical concepts:
    1. Individual Concepts: Specific entities (patients, diseases, drugs)
    2. Qualitative Concepts: Categories/classifications (conditions, findings)
    3. Comparative Concepts: Comparisons (better/worse, severity levels)
    4. Quantitative Concepts: Numerical measurements (lab values, vital signs)
    """
    
    # Core concept identification
    concept_id: str = Field(
        description="Unique identifier for the medical concept in kebab-case format"
    )
    
    concept_name: str = Field(
        description="The specific medical concept, term, or entity identified in the text"
    )
    
    concept_type: str = Field(
        description="Type of medical concept: individual, qualitative, comparative, or quantitative"
    )
    
    concept_category: str = Field(
        description="Medical category: disease, medication, symptom, lab_value, vital_sign, procedure, anatomy, patient, treatment, diagnosis, finding, measurement, or other"
    )
    
    # Relationship information for medical knowledge graphs
    related_concept_id: Optional[str] = Field(
        default=None,
        description="ID of the related medical concept this concept connects to"
    )
    
    related_concept_name: Optional[str] = Field(
        default=None,
        description="Name of the related medical concept"
    )
    
    relationship_type: str = Field(
        description="Medical relationship type: increases, decreases, improves, worsens, causes, treats, indicates, contraindicates, is_part_of, influences, measures, diagnoses, or associated_with"
    )
    
    relationship_strength: float = Field(
        default=1.0,
        description="Clinical relevance strength of the relationship (0.0 to 1.0)"
    )
    
    # Clinical context and evidence
    text_context: str = Field(
        description="The specific clinical text where this medical concept and relationship was identified"
    )
    
    confidence_score: float = Field(
        default=0.8,
        description="AI confidence in medical concept identification and clinical relationship extraction (0.0 to 1.0)"
    )
    
    # Additional medical metadata
    medical_specialty: Optional[str] = Field(
        default=None,
        description="Primary medical specialty domain: cardiology, neurology, oncology, internal_medicine, surgery, pediatrics, psychiatry, emergency_medicine, or general_medicine"
    )
    
    clinical_significance: Optional[str] = Field(
        default=None,
        description="Clinical significance level: critical, high, moderate, low, or informational"
    )


class MedicalConceptsExtraction(BaseModel):
    """
    Container model for medical concepts extraction results with clinical metadata.
    """
    
    concepts: List[MedicalConcept] = Field(
        description="List of extracted medical concepts with clinical relationships"
    )
    
    total_concepts: int = Field(
        default=0,
        description="Total number of medical concepts extracted"
    )


def load_medical_text_content(file_path: Optional[str] = None) -> Optional[str]:
    """
    Load medical text content from a file path or Streamlit session state.
    
    Args:
        file_path: Optional path to text file. If None, tries to load from session state.
        
    Returns:
        String containing text content, or None if loading fails
    """
    logger.info("Starting medical text content loading for concept extraction")
    
    # First try to load from file path if provided
    if file_path:
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists() and file_path_obj.is_file():
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"✓ Successfully loaded medical text from file: {file_path}")
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
            logger.info("✓ Successfully loaded medical text from session state")
            logger.info(f"Content length: {len(content)} characters")
            return content
        else:
            logger.warning("No medical text content found in session state")
    except Exception as e:
        logger.error(f"Error accessing session state: {e}")
    
    logger.error("Failed to load medical text content from any source")
    return None


def extract_medical_concepts(text_content: str, chunk_size: int = 4000) -> Optional[MedicalConceptsExtraction]:
    """
    Extract medical concepts and clinical relationships from text using AI analysis.
    
    This function processes medical texts to identify specific medical concepts according to
    the four-category medical concept framework and their clinical relationships.
    After extraction, it filters to keep only the most relevant and frequent concepts (max 50).
    
    Args:
        text_content: The medical text content to analyze
        chunk_size: Maximum size of text chunks for processing (default: 4000 characters)
        
    Returns:
        MedicalConceptsExtraction object with filtered medical concepts and clinical relationships, or None if extraction fails
    """
    logger.info("Starting AI-powered medical concept extraction with relevance filtering")
    
    if not client:
        logger.error("Gemini client not available for concept extraction")
        return None
    
    if not text_content or len(text_content.strip()) < 50:
        logger.error("Insufficient text content for concept extraction")
        return None
    
    # Process text in chunks to handle large documents
    text_chunks = _split_text_into_chunks(text_content, chunk_size)
    all_concepts = []
    
    for i, chunk in enumerate(text_chunks):
        logger.info(f"Processing text chunk {i+1}/{len(text_chunks)}")
        
        concepts_prompt = f"""
        You are an expert medical knowledge extractor specialized in clinical concept identification and medical knowledge graphs.
        
        Analyze the following medical text and extract SPECIFIC MEDICAL CONCEPTS according to these four categories:
        
        1. INDIVIDUAL CONCEPTS: Specific entities like:
           - Specific diseases: "acute myocardial infarction", "type 2 diabetes mellitus"
           - Specific medications: "metformin 500mg", "lisinopril 10mg"
           - Specific patients: "45-year-old male", "elderly woman"
           - Specific anatomical structures: "left anterior descending artery", "right kidney"
        
        2. QUALITATIVE CONCEPTS: Medical classifications and categories like:
           - Condition categories: "hypertension", "bacterial infection", "malignancy"
           - Clinical findings: "positive blood culture", "elevated white count", "abnormal ECG"
           - Assessment categories: "stable condition", "acute presentation", "chronic disease"
        
        3. COMPARATIVE CONCEPTS: Medical comparisons like:
           - Temporal comparisons: "worse than yesterday", "improved since last visit"
           - Severity comparisons: "more severe than previous episode", "mild compared to baseline"
           - Reference comparisons: "above normal range", "within therapeutic levels"
        
        4. QUANTITATIVE CONCEPTS: Numerical medical measurements like:
           - Vital signs: "blood pressure 140/90 mmHg", "heart rate 85 bpm", "temperature 38.5°C"
           - Lab values: "hemoglobin 8.2 g/dL", "glucose 180 mg/dL", "creatinine 2.1 mg/dL"
           - Measurements: "tumor size 3.2 cm", "ejection fraction 45%"
        
        MEDICAL RELATIONSHIPS to identify:
        - increases/decreases (e.g., "medication increases blood pressure")
        - improves/worsens (e.g., "treatment improves symptoms")
        - causes (e.g., "infection causes fever")
        - treats (e.g., "antibiotic treats pneumonia")
        - indicates (e.g., "elevated troponin indicates heart damage")
        - contraindicates (e.g., "allergy contraindicates medication")
        - is_part_of (e.g., "chest pain is part of heart attack")
        - influences (e.g., "diet influences blood sugar")
        - measures (e.g., "hemoglobin measures anemia")
        - diagnoses (e.g., "CT scan diagnoses stroke")
        - associated_with (e.g., "diabetes associated with neuropathy")
        
        MEDICAL TEXT TO ANALYZE:
        {chunk}
        
        IMPORTANT INSTRUCTIONS:
        - Extract 8-15 clinically relevant medical concepts per chunk
        - Create concept_id in kebab-case format (e.g., "acute-myocardial-infarction", "blood-pressure-140-90")
        - Focus on concepts that have clear clinical relationships
        - Use relationship_strength based on clinical importance (0.9-1.0 for critical, 0.7-0.8 for important, 0.5-0.6 for moderate)
        - Include confidence_score based on clarity of clinical evidence (0.9-1.0 for explicit, 0.7-0.8 for implied, 0.5-0.6 for uncertain)
        - Assign clinical_significance: critical, high, moderate, low, or informational
        - Use only specified concept_type values: individual, qualitative, comparative, quantitative
        - Use only specified concept_category values: disease, medication, symptom, lab_value, vital_sign, procedure, anatomy, patient, treatment, diagnosis, finding, measurement, other
        - Use only specified relationship_type values: increases, decreases, improves, worsens, causes, treats, indicates, contraindicates, is_part_of, influences, measures, diagnoses, associated_with
        - Use only specified medical_specialty values: cardiology, neurology, oncology, internal_medicine, surgery, pediatrics, psychiatry, emergency_medicine, general_medicine
        
        Focus on extracting concepts that would be valuable for clinical decision-making and medical knowledge representation.
        
        Return a comprehensive JSON response with all extracted medical concepts and their clinical relationships.
        """
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=concepts_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": MedicalConceptsExtraction,
                },
            )
            
            # Parse the response and add concepts to collection
            chunk_extraction = response.parsed
            if chunk_extraction and chunk_extraction.concepts:
                all_concepts.extend(chunk_extraction.concepts)
                logger.info(f"✓ Extracted {len(chunk_extraction.concepts)} medical concepts from chunk {i+1}")
            else:
                logger.warning(f"No medical concepts extracted from chunk {i+1}")
                
        except Exception as e:
            logger.error(f"Error extracting medical concepts from chunk {i+1}: {e}")
            continue
    
    if not all_concepts:
        logger.error("No medical concepts were successfully extracted")
        return None
    
    # Filter concepts to select the most relevant and frequent ones (max 50)
    logger.info(f"Filtering {len(all_concepts)} concepts to select most relevant (max 50)")
    filtered_concepts = _filter_most_relevant_concepts(all_concepts, text_content, max_concepts=50)
    
    # Create final extraction result with filtered concepts
    extraction_result = MedicalConceptsExtraction(
        concepts=filtered_concepts,
        total_concepts=len(filtered_concepts)
    )
    
    logger.info(f"✓ Medical concept extraction completed successfully")
    logger.info(f"Total medical concepts extracted: {len(all_concepts)}, filtered to: {extraction_result.total_concepts}")
    
    return extraction_result


def extract_medical_concepts_with_progress(text_content: str, chunk_size: int = 4000, progress_callback=None) -> Optional[MedicalConceptsExtraction]:
    """
    Extract medical concepts and clinical relationships from text using AI analysis with progress tracking.
    
    This function processes medical texts to identify specific medical concepts according to
    the four-category medical concept framework and their clinical relationships.
    After extraction, it filters to keep only the most relevant and frequent concepts (max 50).
    
    Args:
        text_content: The medical text content to analyze
        chunk_size: Maximum size of text chunks for processing (default: 4000 characters)
        progress_callback: Optional callback function to report progress (current_chunk, total_chunks, status)
        
    Returns:
        MedicalConceptsExtraction object with filtered medical concepts and clinical relationships, or None if extraction fails
    """
    logger.info("Starting AI-powered medical concept extraction with progress tracking and relevance filtering")
    
    if not client:
        logger.error("Gemini client not available for concept extraction")
        return None
    
    if not text_content or len(text_content.strip()) < 50:
        logger.error("Insufficient text content for concept extraction")
        return None
    
    # Process text in chunks to handle large documents
    text_chunks = _split_text_into_chunks(text_content, chunk_size)
    total_chunks = len(text_chunks)
    all_concepts = []
    
    # Report initial progress
    if progress_callback:
        progress_callback(0, total_chunks, "Initializing medical concept extraction...")
    
    for i, chunk in enumerate(text_chunks):
        current_chunk = i + 1
        logger.info(f"Processing text chunk {current_chunk}/{total_chunks}")
        
        # Update progress
        if progress_callback:
            progress_callback(current_chunk, total_chunks, f"Extracting concepts from text chunk {current_chunk}")
        
        concepts_prompt = f"""
        You are an expert medical knowledge extractor specialized in clinical concept identification and medical knowledge graphs.
        
        Analyze the following medical text and extract SPECIFIC MEDICAL CONCEPTS according to these four categories:
        
        1. INDIVIDUAL CONCEPTS: Specific entities like:
           - Specific diseases: "acute myocardial infarction", "type 2 diabetes mellitus"
           - Specific medications: "metformin 500mg", "lisinopril 10mg"
           - Specific patients: "45-year-old male", "elderly woman"
           - Specific anatomical structures: "left anterior descending artery", "right kidney"
        
        2. QUALITATIVE CONCEPTS: Medical classifications and categories like:
           - Condition categories: "hypertension", "bacterial infection", "malignancy"
           - Clinical findings: "positive blood culture", "elevated white count", "abnormal ECG"
           - Assessment categories: "stable condition", "acute presentation", "chronic disease"
        
        3. COMPARATIVE CONCEPTS: Medical comparisons like:
           - Temporal comparisons: "worse than yesterday", "improved since last visit"
           - Severity comparisons: "more severe than previous episode", "mild compared to baseline"
           - Reference comparisons: "above normal range", "within therapeutic levels"
        
        4. QUANTITATIVE CONCEPTS: Numerical medical measurements like:
           - Vital signs: "blood pressure 140/90 mmHg", "heart rate 85 bpm", "temperature 38.5°C"
           - Lab values: "hemoglobin 8.2 g/dL", "glucose 180 mg/dL", "creatinine 2.1 mg/dL"
           - Measurements: "tumor size 3.2 cm", "ejection fraction 45%"
        
        MEDICAL RELATIONSHIPS to identify:
        - increases/decreases (e.g., "medication increases blood pressure")
        - improves/worsens (e.g., "treatment improves symptoms")
        - causes (e.g., "infection causes fever")
        - treats (e.g., "antibiotic treats pneumonia")
        - indicates (e.g., "elevated troponin indicates heart damage")
        - contraindicates (e.g., "allergy contraindicates medication")
        - is_part_of (e.g., "chest pain is part of heart attack")
        - influences (e.g., "diet influences blood sugar")
        - measures (e.g., "hemoglobin measures anemia")
        - diagnoses (e.g., "CT scan diagnoses stroke")
        - associated_with (e.g., "diabetes associated with neuropathy")
        
        MEDICAL TEXT TO ANALYZE:
        {chunk}
        
        IMPORTANT INSTRUCTIONS:
        - Extract 8-15 clinically relevant medical concepts per chunk
        - Create concept_id in kebab-case format (e.g., "acute-myocardial-infarction", "blood-pressure-140-90")
        - Focus on concepts that have clear clinical relationships
        - Use relationship_strength based on clinical importance (0.9-1.0 for critical, 0.7-0.8 for important, 0.5-0.6 for moderate)
        - Include confidence_score based on clarity of clinical evidence (0.9-1.0 for explicit, 0.7-0.8 for implied, 0.5-0.6 for uncertain)
        - Assign clinical_significance: critical, high, moderate, low, or informational
        - Use only specified concept_type values: individual, qualitative, comparative, quantitative
        - Use only specified concept_category values: disease, medication, symptom, lab_value, vital_sign, procedure, anatomy, patient, treatment, diagnosis, finding, measurement, other
        - Use only specified relationship_type values: increases, decreases, improves, worsens, causes, treats, indicates, contraindicates, is_part_of, influences, measures, diagnoses, associated_with
        - Use only specified medical_specialty values: cardiology, neurology, oncology, internal_medicine, surgery, pediatrics, psychiatry, emergency_medicine, general_medicine
        
        Focus on extracting concepts that would be valuable for clinical decision-making and medical knowledge representation.
        
        Return a comprehensive JSON response with all extracted medical concepts and their clinical relationships.
        """
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=concepts_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": MedicalConceptsExtraction,
                },
            )
            
            # Parse the response and add concepts to collection
            chunk_extraction = response.parsed
            if chunk_extraction and chunk_extraction.concepts:
                all_concepts.extend(chunk_extraction.concepts)
                logger.info(f"✓ Extracted {len(chunk_extraction.concepts)} medical concepts from chunk {current_chunk}")
                
                # Update progress with extraction results
                if progress_callback:
                    progress_callback(current_chunk, total_chunks, 
                                    f"Extracted {len(chunk_extraction.concepts)} concepts from chunk {current_chunk}")
            else:
                logger.warning(f"No medical concepts extracted from chunk {current_chunk}")
                if progress_callback:
                    progress_callback(current_chunk, total_chunks, 
                                    f"No concepts found in chunk {current_chunk}")
                
        except Exception as e:
            logger.error(f"Error extracting medical concepts from chunk {current_chunk}: {e}")
            if progress_callback:
                progress_callback(current_chunk, total_chunks, 
                                f"Error processing chunk {current_chunk}: {str(e)}")
            continue
    
    if not all_concepts:
        logger.error("No medical concepts were successfully extracted")
        if progress_callback:
            progress_callback(total_chunks, total_chunks, "No medical concepts were extracted")
        return None
    
    # Update progress for filtering phase
    if progress_callback:
        progress_callback(total_chunks, total_chunks, f"Filtering {len(all_concepts)} concepts to select most relevant...")
    
    # Filter concepts to select the most relevant and frequent ones (max 50)
    logger.info(f"Filtering {len(all_concepts)} concepts to select most relevant (max 50)")
    filtered_concepts = _filter_most_relevant_concepts(all_concepts, text_content, max_concepts=50)
    
    # Create final extraction result with filtered concepts
    extraction_result = MedicalConceptsExtraction(
        concepts=filtered_concepts,
        total_concepts=len(filtered_concepts)
    )
    
    # Final progress update
    if progress_callback:
        progress_callback(total_chunks, total_chunks, 
                        f"Completed! Extracted {len(all_concepts)} concepts, filtered to {extraction_result.total_concepts}")
    
    logger.info(f"✓ Medical concept extraction completed successfully")
    logger.info(f"Total medical concepts extracted: {len(all_concepts)}, filtered to: {extraction_result.total_concepts}")
    
    return extraction_result


def _filter_most_relevant_concepts(concepts: List[MedicalConcept], text_content: str, max_concepts: int = 50) -> List[MedicalConcept]:
    """
    Filter medical concepts to select the most relevant and frequent ones.
    
    This function analyzes concepts based on:
    1. Frequency of concept name appearance in the original text
    2. Clinical significance level
    3. Confidence score
    4. Relationship strength
    5. Uniqueness (to avoid duplicates)
    
    Args:
        concepts: List of extracted medical concepts
        text_content: Original text content for frequency analysis
        max_concepts: Maximum number of concepts to return
        
    Returns:
        List of filtered medical concepts, ordered by relevance
    """
    logger.info(f"Starting concept filtering process with {len(concepts)} concepts")
    
    if len(concepts) <= max_concepts:
        logger.info("Number of concepts is already within limit, returning all concepts")
        return concepts
    
    # Convert text to lowercase for case-insensitive frequency analysis
    text_lower = text_content.lower()
    
    # Calculate relevance score for each concept
    concept_scores = []
    unique_concept_names = set()  # Track unique concept names to avoid duplicates
    
    for concept in concepts:
        # Skip duplicate concept names (keep the one with higher confidence)
        concept_name_lower = concept.concept_name.lower()
        if concept_name_lower in unique_concept_names:
            # Check if this concept has higher confidence than existing one
            existing_concept = next((c for c in concept_scores if c['concept'].concept_name.lower() == concept_name_lower), None)
            if existing_concept and concept.confidence_score > existing_concept['concept'].confidence_score:
                # Replace with higher confidence concept
                concept_scores.remove(existing_concept)
                unique_concept_names.discard(concept_name_lower)
            else:
                continue  # Skip this lower confidence duplicate
        
        unique_concept_names.add(concept_name_lower)
        
        # Calculate frequency score for each concept
        # Look for exact matches and partial matches of the concept name
        concept_words = concept.concept_name.lower().split()
        frequency_score = 0
        
        # Count exact phrase occurrences
        exact_matches = text_lower.count(concept.concept_name.lower())
        frequency_score += exact_matches * 3  # Weight exact matches more heavily
        
        # Count individual word occurrences (for multi-word concepts)
        for word in concept_words:
            if len(word) > 3:  # Only count meaningful words
                word_matches = text_lower.count(word)
                frequency_score += word_matches * 0.5
        
        # Calculate clinical significance score
        significance_scores = {
            'critical': 5.0,
            'high': 4.0,
            'moderate': 3.0,
            'low': 2.0,
            'informational': 1.0
        }
        significance_score = significance_scores.get(concept.clinical_significance, 2.5)
        
        # Calculate concept type priority (prioritize clinical concepts)
        type_priorities = {
            'individual': 4.0,      # Specific entities are highly valuable
            'quantitative': 3.5,    # Measurements are very important
            'qualitative': 3.0,     # Classifications are important
            'comparative': 2.5      # Comparisons are moderately important
        }
        type_score = type_priorities.get(concept.concept_type, 2.0)
        
        # Calculate category priority (prioritize clinical categories)
        category_priorities = {
            'disease': 4.5,
            'medication': 4.0,
            'procedure': 3.5,
            'symptom': 3.5,
            'lab_value': 3.5,
            'vital_sign': 3.5,
            'diagnosis': 4.0,
            'treatment': 3.5,
            'finding': 3.0,
            'anatomy': 2.5,
            'measurement': 3.0,
            'patient': 2.0,
            'other': 1.0
        }
        category_score = category_priorities.get(concept.concept_category, 2.0)
        
        # Calculate overall relevance score
        # Weighted combination of different factors
        relevance_score = (
            frequency_score * 0.3 +           # 30% frequency in text
            significance_score * 0.25 +       # 25% clinical significance
            concept.confidence_score * 0.2 +  # 20% AI confidence
            type_score * 0.15 +               # 15% concept type priority
            category_score * 0.1              # 10% category priority
        )
        
        # Add bonus for concepts with relationships
        if concept.related_concept_id and concept.relationship_strength:
            relevance_score += concept.relationship_strength * 0.5
        
        concept_scores.append({
            'concept': concept,
            'relevance_score': relevance_score,
            'frequency_score': frequency_score,
            'significance_score': significance_score
        })
    
    # Sort concepts by relevance score (highest first)
    concept_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Select top concepts up to max_concepts
    selected_concepts = [item['concept'] for item in concept_scores[:max_concepts]]
    
    # Log filtering results
    logger.info(f"✓ Filtered concepts from {len(concepts)} to {len(selected_concepts)}")
    if concept_scores:
        highest_score = concept_scores[0]['relevance_score']
        lowest_selected_score = concept_scores[min(len(concept_scores)-1, max_concepts-1)]['relevance_score']
        logger.info(f"Relevance score range: {lowest_selected_score:.2f} to {highest_score:.2f}")
    
    # Log top 5 concepts for debugging
    logger.info("Top 5 selected concepts:")
    for i, item in enumerate(concept_scores[:5]):
        concept = item['concept']
        score = item['relevance_score']
        logger.info(f"  {i+1}. {concept.concept_name} (score: {score:.2f}, freq: {item['frequency_score']:.1f}, sig: {concept.clinical_significance})")
    
    return selected_concepts


def _split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Split text into manageable chunks for processing while preserving sentence boundaries.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences to maintain context
    sentences = text.split('. ')
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info(f"Text split into {len(chunks)} chunks for processing")
    return chunks


def save_concepts_to_json(extraction_result: MedicalConceptsExtraction, original_filename: str = "medical_text") -> Optional[str]:
    """
    Save the extracted medical concepts to a JSON file optimized for medical knowledge graphs and Neo4j import.
    
    Args:
        extraction_result: The extraction results containing medical concepts and clinical relationships
        original_filename: Base filename for the concepts file
        
    Returns:
        Path to saved medical concepts file, or None if saving fails
    """
    logger.info("Starting medical concepts JSON file save process")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate medical concepts filename
        base_name = Path(original_filename).stem
        concepts_filename = f"{base_name}_medical_concepts.json"
        concepts_path = output_dir / concepts_filename
        
        # Create enhanced JSON structure with medical metadata
        enhanced_data = {
            "timestamp_created": datetime.datetime.now().isoformat(),
            "extraction_metadata": {
                "source_file": original_filename,
                "total_concepts": extraction_result.total_concepts,
                "total_relationships": len([c for c in extraction_result.concepts if c.related_concept_id]),
                "extraction_model": "gemini-2.0-flash",
                "filtering_applied": "top_50_most_relevant" if extraction_result.total_concepts <= 50 else "none",
                "concept_types_found": list(set(c.concept_type for c in extraction_result.concepts)),
                "concept_categories_found": list(set(c.concept_category for c in extraction_result.concepts)),
                "relationship_types_found": list(set(c.relationship_type for c in extraction_result.concepts)),
                "medical_specialties_found": list(set(c.medical_specialty for c in extraction_result.concepts if c.medical_specialty)),
                "clinical_significance_levels": list(set(c.clinical_significance for c in extraction_result.concepts if c.clinical_significance))
            },
            "medical_concepts": [concept.model_dump() for concept in extraction_result.concepts],
            "total_concepts": extraction_result.total_concepts,
            "schema_info": {
                "concept_types": ["individual", "qualitative", "comparative", "quantitative"],
                "concept_categories": ["disease", "medication", "symptom", "lab_value", "vital_sign", "procedure", "anatomy", "patient", "treatment", "diagnosis", "finding", "measurement", "other"],
                "relationship_types": ["increases", "decreases", "improves", "worsens", "causes", "treats", "indicates", "contraindicates", "is_part_of", "influences", "measures", "diagnoses", "associated_with"],
                "medical_specialties": ["cardiology", "neurology", "oncology", "internal_medicine", "surgery", "pediatrics", "psychiatry", "emergency_medicine", "general_medicine"],
                "clinical_significance": ["critical", "high", "moderate", "low", "informational"],
                "filtering_info": {
                    "max_concepts": 50,
                    "filtering_criteria": [
                        "frequency_in_text (30%)",
                        "clinical_significance (25%)",
                        "ai_confidence_score (20%)",
                        "concept_type_priority (15%)",
                        "category_priority (10%)",
                        "relationship_bonus (variable)"
                    ]
                }
            }
        }
        
        # Write medical concepts to JSON file with pretty formatting
        with open(concepts_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Medical concepts saved successfully: {concepts_path}")
        return str(concepts_path)
        
    except Exception as e:
        logger.error(f"Error saving medical concepts file: {e}")
        return None


def display_extraction_results(extraction_result: MedicalConceptsExtraction) -> None:
    """
    Display medical concept extraction results in a clinically-focused format in the terminal.
    
    Args:
        extraction_result: The medical concept extraction results to display
    """
    print("\n" + "="*80)
    print("🏥 MEDICAL CONCEPTS EXTRACTION RESULTS (FILTERED)")
    print("="*80)
    
    print(f"📊 CLINICAL SUMMARY:")
    print(f"   Total Medical Concepts Extracted: {extraction_result.total_concepts}")
    print(f"   Total Clinical Relationships: {len([c for c in extraction_result.concepts if c.related_concept_id])}")
    print(f"   Filtering Applied: Most relevant concepts selected (max 50)")
    
    # Group concepts by medical type
    concept_types = {}
    for concept in extraction_result.concepts:
        if concept.concept_type not in concept_types:
            concept_types[concept.concept_type] = []
        concept_types[concept.concept_type].append(concept)
    
    print(f"\n🧬 MEDICAL CONCEPT TYPES DISTRIBUTION:")
    for concept_type, concepts in concept_types.items():
        print(f"   {concept_type.replace('_', ' ').title()}: {len(concepts)}")
    
    # Group concepts by medical category
    concept_categories = {}
    for concept in extraction_result.concepts:
        if concept.concept_category not in concept_categories:
            concept_categories[concept.concept_category] = []
        concept_categories[concept.concept_category].append(concept)
    
    print(f"\n📋 MEDICAL CATEGORIES DISTRIBUTION:")
    for category, concepts in concept_categories.items():
        print(f"   {category.replace('_', ' ').title()}: {len(concepts)}")
    
    # Display sample medical concepts with clinical details
    print(f"\n🔬 TOP MEDICAL CONCEPTS (MOST RELEVANT):")
    for i, concept in enumerate(extraction_result.concepts[:10]):  # Show first 10 medical concepts
        print(f"\n   [{i+1}] {concept.concept_name}")
        print(f"       ID: {concept.concept_id}")
        print(f"       Type: {concept.concept_type}")
        print(f"       Category: {concept.concept_category}")
        print(f"       Specialty: {concept.medical_specialty or 'General Medicine'}")
        print(f"       Clinical Significance: {concept.clinical_significance or 'Not specified'}")
        if concept.related_concept_name:
            print(f"       Clinical Relationship: {concept.relationship_type} → {concept.related_concept_name}")
            print(f"       Relationship Strength: {concept.relationship_strength:.2f}")
        print(f"       Confidence: {concept.confidence_score:.2f}")
        print(f"       Clinical Context: {concept.text_context[:120]}...")
    
    if len(extraction_result.concepts) > 10:
        print(f"\n   ... and {len(extraction_result.concepts) - 10} more filtered medical concepts")
    
    # Show clinical relationship types
    relationship_types = set(c.relationship_type for c in extraction_result.concepts)
    print(f"\n🔗 CLINICAL RELATIONSHIP TYPES FOUND:")
    for rel_type in sorted(relationship_types):
        count = len([c for c in extraction_result.concepts if c.relationship_type == rel_type])
        print(f"   {rel_type}: {count}")
    
    # Show medical specialties
    specialties = set(c.medical_specialty for c in extraction_result.concepts if c.medical_specialty)
    if specialties:
        print(f"\n🏥 MEDICAL SPECIALTIES IDENTIFIED:")
        for specialty in sorted(specialties):
            count = len([c for c in extraction_result.concepts if c.medical_specialty == specialty])
            print(f"   {specialty.replace('_', ' ').title()}: {count}")
    
    # Show clinical significance levels
    significance_levels = set(c.clinical_significance for c in extraction_result.concepts if c.clinical_significance)
    if significance_levels:
        print(f"\n⚠️  CLINICAL SIGNIFICANCE LEVELS:")
        for level in sorted(significance_levels, key=lambda x: ["critical", "high", "moderate", "low", "informational"].index(x) if x in ["critical", "high", "moderate", "low", "informational"] else 999):
            count = len([c for c in extraction_result.concepts if c.clinical_significance == level])
            print(f"   {level.title()}: {count}")
    
    print(f"\n🎯 FILTERING SUMMARY:")
    print(f"   Concepts are ranked by relevance using:")
    print(f"   • Text frequency (30%) - How often the concept appears")
    print(f"   • Clinical significance (25%) - Medical importance level")
    print(f"   • AI confidence (20%) - Extraction confidence score")
    print(f"   • Concept type priority (15%) - Clinical value of concept type")
    print(f"   • Category priority (10%) - Medical category importance")
    print(f"   • Relationship bonus - Extra points for connected concepts")
    
    print("\n" + "="*80)


def process_medical_text_for_concepts(file_path: Optional[str] = None) -> tuple[bool, Optional[str], Optional[MedicalConceptsExtraction]]:
    """
    Complete workflow to process medical text for concept extraction and Neo4j preparation.
    
    Args:
        file_path: Optional path to text file. If None, loads from session state.
        
    Returns:
        Tuple of (success_flag, concepts_file_path, extraction_results)
    """
    logger.info("=== Starting complete medical concept extraction workflow ===")
    
    try:
        # Step 1: Load medical text content
        logger.info("Step 1: Loading medical text content")
        content = load_medical_text_content(file_path)
        if not content:
            logger.error("Failed to load medical text content")
            return False, None, None
        
        # Step 2: Extract medical concepts using AI
        logger.info("Step 2: Extracting medical concepts and relationships")
        extraction_result = extract_medical_concepts(content)
        if not extraction_result:
            logger.error("Failed to extract medical concepts")
            return False, None, None
        
        # Step 3: Display results in terminal
        logger.info("Step 3: Displaying extraction results")
        display_extraction_results(extraction_result)
        
        # Step 4: Save concepts to JSON file for Neo4j import
        logger.info("Step 4: Saving concepts to JSON file")
        original_name = file_path if file_path else "session_document"
        concepts_path = save_concepts_to_json(extraction_result, original_name)
        if not concepts_path:
            logger.error("Failed to save concepts file")
            return False, None, extraction_result
        
        logger.info("=== Medical concept extraction workflow completed successfully ===")
        return True, concepts_path, extraction_result
        
    except Exception as e:
        logger.error(f"Unexpected error in medical concept extraction workflow: {e}")
        return False, None, None


def test_concepts_agent():
    """
    Test the medical concepts extraction agent using the Self-directed learning journal article.
    
    This test demonstrates the complete workflow for extracting medical concepts
    from a real medical research paper with focus on clinical relevance.
    """
    logger.info("=== Running Medical Concepts Extraction Agent Tests ===")
    
    # Test 1: Check dependencies and setup
    logger.info("Test 1: Checking dependencies and AI client setup")
    try:
        from google import genai
        from pydantic import BaseModel
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
    
    # Test 3: Load the Self-directed learning journal article
    logger.info("Test 3: Loading Self-directed learning journal article for testing")
    test_file_path = Path("output/Self-directed learning in health professions.md")
    
    if not test_file_path.exists():
        logger.error(f"✗ Test file not found: {test_file_path}")
        logger.info("Please ensure the journal article file is available for testing")
        return False
    
    content = load_medical_text_content(str(test_file_path))
    if not content:
        logger.error("✗ Failed to load journal article content")
        return False
    
    logger.info(f"✓ Journal article loaded successfully ({len(content)} characters)")
    
    # Test 4: Extract medical concepts from the journal article
    logger.info("Test 4: Testing medical concept extraction from journal article")
    extraction_result = extract_medical_concepts(content[:8000])  # Use first 8000 chars for testing
    if not extraction_result:
        logger.error("✗ Failed to extract medical concepts")
        return False
    
    logger.info("✓ Medical concept extraction completed successfully")
    
    # Test 5: Display clinical results
    logger.info("Test 5: Displaying medical extraction results")
    display_extraction_results(extraction_result)
    
    # Test 6: Save medical concepts to JSON file
    logger.info("Test 6: Testing medical concepts JSON file save functionality")
    concepts_path = save_concepts_to_json(extraction_result, "Self-directed_learning_medical_concepts")
    if not concepts_path:
        logger.error("✗ Failed to save medical concepts file")
        return False
    
    logger.info(f"✓ Medical concepts saved successfully: {concepts_path}")
    
    # Test 7: Verify saved medical JSON structure
    logger.info("Test 7: Verifying saved medical concepts JSON file structure")
    try:
        with open(concepts_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        logger.info(f"✓ Medical concepts JSON file verified with {len(saved_data.get('medical_concepts', []))} concepts")
        logger.info(f"File size: {os.path.getsize(concepts_path)} bytes")
        
        # Display JSON structure for medical knowledge graph preparation
        print(f"\n🏥 MEDICAL KNOWLEDGE GRAPH JSON STRUCTURE:")
        print(f"   File: {concepts_path}")
        print(f"   Structure contains:")
        print(f"     - medical_concepts: List of {len(saved_data.get('medical_concepts', []))} clinical concepts")
        print(f"     - extraction_metadata: Clinical analysis metadata")
        print(f"     - schema_info: Medical ontology structure")
        print(f"     - total_concepts: {saved_data.get('total_concepts', 0)}")
        print(f"   Ready for Medical Knowledge Graph import using:")
        print(f"     - concept_id as medical node identifier")
        print(f"     - relationship_type for clinical edge types")
        print(f"     - relationship_strength for clinical relevance weighting")
        print(f"     - clinical_significance for priority indexing")
        print(f"     - medical_specialty for domain-specific querying")
        
    except Exception as e:
        logger.error(f"✗ Error verifying medical concepts JSON file: {e}")
        return False
    
    # Test 8: Complete medical workflow test
    logger.info("Test 8: Testing complete medical concept extraction workflow")
    success, workflow_concepts_path, workflow_extraction = process_medical_text_for_concepts(str(test_file_path))
    
    if success:
        logger.info("✓ Complete medical workflow test passed")
        logger.info(f"Medical workflow concepts path: {workflow_concepts_path}")
    else:
        logger.error("✗ Complete medical workflow test failed")
        return False
    
    logger.info("=== All Medical Concepts Extraction Tests Passed! ===")
    logger.info(f"Test completed using medical journal article: {test_file_path}")
    print(f"\n🎉 SUCCESS! The medical concepts extraction system successfully:")
    print(f"   ✓ Loaded medical research text content")
    print(f"   ✓ Identified specific medical concepts by type (individual, qualitative, comparative, quantitative)")
    print(f"   ✓ Categorized medical concepts (diseases, medications, symptoms, lab values, etc.)")
    print(f"   ✓ Extracted clinical relationships with medical relevance")
    print(f"   ✓ Generated Medical Knowledge Graph-ready JSON structure")
    print(f"   ✓ Saved structured data optimized for clinical decision support")
    print(f"\n📋 Next steps for Medical Knowledge Graph integration:")
    print(f"   1. Import medical concepts as nodes using concept_id")
    print(f"   2. Create clinical relationships using relationship_type")
    print(f"   3. Apply relationship_strength as clinical relevance weights")
    print(f"   4. Index by concept_category and medical_specialty")
    print(f"   5. Enable clinical queries using clinical_significance levels")
    print(f"   6. Support differential diagnosis through concept relationships")
    
    return True


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Run comprehensive tests with the Self-directed learning journal article
    test_concepts_agent()