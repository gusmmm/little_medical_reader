import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import streamlit as st
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Configure logging for this module
logger = logging.getLogger(__name__)

@dataclass
class TextStatistics:
    """
    Statistical information about the analyzed text.
    """
    total_characters: int
    total_words: int
    total_sentences: int
    total_paragraphs: int
    average_words_per_sentence: float
    average_sentences_per_paragraph: float
    reading_level: str

@dataclass
class TerminologyExtraction:
    """
    Medical and scientific terminology extracted from the text.
    """
    medical_terms: List[str]
    scientific_terms: List[str]
    methodology_terms: List[str]
    statistical_terms: List[str]
    research_terms: List[str]

class SemanticStructure(BaseModel):
    """
    Semantic structure analysis of the medical/scientific text using Pydantic v2.
    """
    document_sections: List[str] = Field(description="Main document sections identified")
    research_methodology: Optional[str] = Field(default=None, description="Research methodology identified")
    study_population: Optional[str] = Field(default=None, description="Study population or sample")
    key_findings: List[str] = Field(default_factory=list, description="Key findings and results")
    clinical_implications: List[str] = Field(default_factory=list, description="Clinical implications mentioned")
    limitations: List[str] = Field(default_factory=list, description="Study limitations identified")

class MedicalNLPAnalysis(BaseModel):
    """
    Comprehensive NLP analysis results for medical and scientific texts using Pydantic v2.
    """
    text_statistics: Dict[str, Any] = Field(description="Basic text statistics")
    terminology_extraction: Dict[str, List[str]] = Field(description="Extracted medical and scientific terminology")
    semantic_structure: SemanticStructure = Field(description="Semantic structure analysis")
    named_entities: Dict[str, List[str]] = Field(description="Named entities categorized by type")
    key_phrases: List[str] = Field(description="Important phrases and collocations")
    readability_metrics: Dict[str, float] = Field(description="Text readability and complexity metrics")
    document_classification: Dict[str, str] = Field(description="Document type and domain classification")

def load_text_content(file_path: Optional[str] = None) -> Optional[str]:
    """
    Load text content from a file path or from Streamlit session state.
    
    Args:
        file_path: Optional path to text file. If None, tries to load from session state.
        
    Returns:
        String containing text content, or None if loading fails
    """
    logger.info("Starting text content loading for NLP analysis")
    
    # First try to load from file path if provided
    if file_path:
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists() and file_path_obj.is_file():
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"✓ Successfully loaded text from file: {file_path}")
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
            logger.info("✓ Successfully loaded text from session state")
            logger.info(f"Content length: {len(content)} characters")
            return content
        else:
            logger.warning("No text content found in session state")
    except Exception as e:
        logger.error(f"Error accessing session state: {e}")
    
    logger.error("Failed to load text content from any source")
    return None

def extract_text_statistics(text: str) -> TextStatistics:
    """
    Extract basic statistical information from the text.
    
    Args:
        text: The input text to analyze
        
    Returns:
        TextStatistics object containing basic text metrics
    """
    logger.info("Extracting basic text statistics")
    
    # Remove markdown formatting for accurate statistics
    clean_text = _clean_markdown(text)
    
    # Count basic elements
    total_characters = len(clean_text)
    
    # Split into words (handle medical abbreviations and hyphenated terms)
    words = re.findall(r'\b[A-Za-z]+(?:[-.]?[A-Za-z]+)*\b', clean_text)
    total_words = len(words)
    
    # Split into sentences (handle medical citations and abbreviations)
    sentences = re.split(r'[.!?]+(?:\s+|$)', clean_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    total_sentences = len(sentences)
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
    total_paragraphs = len(paragraphs)
    
    # Calculate averages
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
    avg_sentences_per_paragraph = total_sentences / total_paragraphs if total_paragraphs > 0 else 0
    
    # Estimate reading level based on sentence complexity
    reading_level = _estimate_reading_level(avg_words_per_sentence)
    
    logger.info(f"Text statistics: {total_words} words, {total_sentences} sentences, {total_paragraphs} paragraphs")
    
    return TextStatistics(
        total_characters=total_characters,
        total_words=total_words,
        total_sentences=total_sentences,
        total_paragraphs=total_paragraphs,
        average_words_per_sentence=avg_words_per_sentence,
        average_sentences_per_paragraph=avg_sentences_per_paragraph,
        reading_level=reading_level
    )

def extract_medical_terminology(text: str) -> TerminologyExtraction:
    """
    Extract medical and scientific terminology from the text using pattern matching and domain knowledge.
    
    Args:
        text: The input text to analyze for terminology
        
    Returns:
        TerminologyExtraction object containing categorized terminology
    """
    logger.info("Starting medical and scientific terminology extraction")
    
    # Clean text for analysis
    clean_text = _clean_markdown(text.lower())
    
    # Define medical terminology patterns
    medical_patterns = {
        'medical_terms': [
            r'\b(?:patient|clinical|medical|health|healthcare|physician|nurse|doctor|treatment|therapy|diagnosis|symptom|disease|condition|disorder|syndrome|pathology|anatomy|physiology)\w*\b',
            r'\b(?:hospital|clinic|ward|intensive care|emergency|surgery|surgical|operative|procedure|intervention|medication|pharmaceutical|drug|dosage)\w*\b',
            r'\b(?:cardiovascular|respiratory|neurological|psychiatric|orthopedic|pediatric|geriatric|oncology|radiology|pathology|pharmacology)\w*\b'
        ],
        'scientific_terms': [
            r'\b(?:research|study|investigation|analysis|methodology|systematic|meta-analysis|randomized|controlled|trial|experiment|hypothesis)\w*\b',
            r'\b(?:data|sample|population|cohort|participant|subject|variable|outcome|measure|assessment|evaluation|validation)\w*\b',
            r'\b(?:evidence|findings|results|conclusion|recommendation|implication|significance|correlation|association)\w*\b'
        ],
        'methodology_terms': [
            r'\b(?:qualitative|quantitative|mixed-methods|cross-sectional|longitudinal|prospective|retrospective|observational)\w*\b',
            r'\b(?:survey|questionnaire|interview|focus group|observation|case study|literature review|systematic review)\w*\b',
            r'\b(?:sampling|recruitment|inclusion|exclusion|criteria|protocol|procedure|intervention|control|placebo)\w*\b'
        ],
        'statistical_terms': [
            r'\b(?:statistical|significance|p-value|confidence interval|odds ratio|correlation|regression|anova|chi-square)\w*\b',
            r'\b(?:mean|median|mode|standard deviation|variance|distribution|normal|skewed|sample size|power)\w*\b',
            r'\b(?:reliability|validity|sensitivity|specificity|precision|accuracy|bias|confounding|adjustment)\w*\b'
        ],
        'research_terms': [
            r'\b(?:literature|publication|journal|article|paper|manuscript|citation|reference|bibliography)\w*\b',
            r'\b(?:theory|model|framework|concept|construct|dimension|factor|component|element)\w*\b',
            r'\b(?:learning|education|training|development|competence|skill|knowledge|understanding|expertise)\w*\b'
        ]
    }
    
    # Extract terms for each category
    extracted_terms = {}
    for category, patterns in medical_patterns.items():
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, clean_text)
            terms.update(matches)
        extracted_terms[category] = sorted(list(terms))
        logger.info(f"Extracted {len(extracted_terms[category])} {category}")
    
    return TerminologyExtraction(
        medical_terms=extracted_terms['medical_terms'],
        scientific_terms=extracted_terms['scientific_terms'],
        methodology_terms=extracted_terms['methodology_terms'],
        statistical_terms=extracted_terms['statistical_terms'],
        research_terms=extracted_terms['research_terms']
    )

def analyze_semantic_structure(text: str) -> SemanticStructure:
    """
    Analyze the semantic structure of medical and scientific text to identify key components.
    
    Args:
        text: The input text to analyze for semantic structure
        
    Returns:
        SemanticStructure object containing structural analysis
    """
    logger.info("Starting semantic structure analysis")
    
    # Identify document sections using common academic/medical patterns
    sections = _identify_document_sections(text)
    
    # Extract research methodology information
    methodology = _extract_research_methodology(text)
    
    # Extract study population information
    study_population = _extract_study_population(text)
    
    # Extract key findings
    key_findings = _extract_key_findings(text)
    
    # Extract clinical implications
    clinical_implications = _extract_clinical_implications(text)
    
    # Extract limitations
    limitations = _extract_limitations(text)
    
    logger.info(f"Identified {len(sections)} document sections and {len(key_findings)} key findings")
    
    return SemanticStructure(
        document_sections=sections,
        research_methodology=methodology,
        study_population=study_population,
        key_findings=key_findings,
        clinical_implications=clinical_implications,
        limitations=limitations
    )

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities relevant to medical and scientific texts using pattern matching.
    
    Args:
        text: The input text to analyze for named entities
        
    Returns:
        Dictionary containing categorized named entities
    """
    logger.info("Starting named entity extraction")
    
    entities = {
        'organizations': [],
        'locations': [],
        'persons': [],
        'medical_conditions': [],
        'medications': [],
        'procedures': [],
        'measurements': [],
        'time_periods': []
    }
    
    # Pattern-based entity extraction
    entity_patterns = {
        'organizations': [
            r'\b(?:University|Hospital|Institute|Centre|Center|Department|College|School|Association|Society|Organization)\s+(?:of\s+)?[A-Z][A-Za-z\s]+\b',
            r'\b[A-Z][A-Za-z\s]*(?:University|Hospital|Institute|Centre|Center|Department|College|School)\b'
        ],
        'locations': [
            r'\b(?:United States|USA|UK|United Kingdom|Canada|Australia|Europe|America|North America|Asia|Africa)\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,}\b'  # City, State/Country
        ],
        'medical_conditions': [
            r'\b(?:diabetes|hypertension|depression|anxiety|cancer|stroke|heart\s+disease|myocardial\s+infarction|pneumonia|infection)\b',
            r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\s+(?:syndrome|disease|disorder|condition|deficiency)\b'
        ],
        'medications': [
            r'\b(?:metformin|insulin|aspirin|lisinopril|atorvastatin|warfarin|furosemide|morphine|acetaminophen|ibuprofen)\b',
            r'\b[A-Z][a-z]+(?:mycin|cillin|olol|pril|statin|ine|ide)\b'
        ],
        'procedures': [
            r'\b(?:surgery|operation|procedure|intervention|therapy|treatment|rehabilitation|consultation|examination|assessment)\b',
            r'\b(?:CT\s+scan|MRI|X-ray|ultrasound|biopsy|endoscopy|catheterization|intubation|dialysis)\b'
        ],
        'measurements': [
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|mmHg|bpm|°C|°F|%|cm|mm|inches|feet)\b',
            r'\b\d+(?:\.\d+)?\s*(?:mg/dL|g/dL|mL/min|beats/min|breaths/min)\b'
        ],
        'time_periods': [
            r'\b(?:daily|weekly|monthly|annually|hourly|twice\s+daily|three\s+times\s+daily)\b',
            r'\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
    }
    
    # Extract entities using patterns
    for entity_type, patterns in entity_patterns.items():
        entity_set = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entity_set.update(matches)
        entities[entity_type] = sorted(list(entity_set))
        logger.info(f"Extracted {len(entities[entity_type])} {entity_type}")
    
    return entities

def extract_key_phrases(text: str, min_length: int = 2, max_length: int = 5) -> List[str]:
    """
    Extract important phrases and collocations from medical and scientific text.
    
    Args:
        text: The input text to analyze
        min_length: Minimum number of words in a phrase
        max_length: Maximum number of words in a phrase
        
    Returns:
        List of key phrases sorted by importance
    """
    logger.info("Starting key phrase extraction")
    
    # Clean and prepare text
    clean_text = _clean_markdown(text.lower())
    
    # Split into sentences and words
    sentences = re.split(r'[.!?]+', clean_text)
    
    # Extract n-grams (phrases of different lengths)
    phrase_counts = Counter()
    
    for sentence in sentences:
        words = re.findall(r'\b[a-z]+(?:[-][a-z]+)*\b', sentence)
        
        # Generate n-grams of different lengths
        for n in range(min_length, max_length + 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                
                # Filter out common phrases and ensure medical/scientific relevance
                if _is_relevant_phrase(phrase):
                    phrase_counts[phrase] += 1
    
    # Sort by frequency and return top phrases
    key_phrases = [phrase for phrase, count in phrase_counts.most_common(50) if count >= 2]
    
    logger.info(f"Extracted {len(key_phrases)} key phrases")
    return key_phrases

def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """
    Calculate various readability and complexity metrics for the text.
    
    Args:
        text: The input text to analyze
        
    Returns:
        Dictionary containing readability metrics
    """
    logger.info("Calculating readability metrics")
    
    clean_text = _clean_markdown(text)
    
    # Basic counts
    words = re.findall(r'\b[A-Za-z]+\b', clean_text)
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    syllables = sum(_count_syllables(word) for word in words)
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = syllables
    
    metrics = {}
    
    if total_sentences > 0 and total_words > 0:
        # Average words per sentence
        avg_words_per_sentence = total_words / total_sentences
        
        # Average syllables per word
        avg_syllables_per_word = total_syllables / total_words
        
        # Flesch Reading Ease Score
        flesch_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
        
        # Medical text complexity (custom metric)
        medical_complexity = _calculate_medical_complexity(text)
        
        metrics = {
            'flesch_reading_ease': max(0, min(100, flesch_ease)),
            'flesch_kincaid_grade': max(0, flesch_kincaid),
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_syllables_per_word': avg_syllables_per_word,
            'medical_complexity_score': medical_complexity,
            'total_words': total_words,
            'total_sentences': total_sentences
        }
    
    logger.info(f"Calculated readability metrics: Flesch={metrics.get('flesch_reading_ease', 0):.1f}")
    return metrics

def classify_document_type(text: str) -> Dict[str, str]:
    """
    Classify the document type and domain based on content analysis.
    
    Args:
        text: The input text to classify
        
    Returns:
        Dictionary containing classification results
    """
    logger.info("Starting document classification")
    
    clean_text = text.lower()
    
    # Document type classification based on structural and content patterns
    document_types = {
        'research_article': ['abstract', 'methodology', 'results', 'discussion', 'conclusion', 'references'],
        'review_article': ['literature review', 'systematic review', 'meta-analysis', 'evidence'],
        'case_report': ['case report', 'case study', 'patient', 'clinical presentation'],
        'clinical_guideline': ['guideline', 'recommendation', 'protocol', 'standard', 'best practice'],
        'technical_report': ['report', 'analysis', 'findings', 'assessment', 'evaluation']
    }
    
    # Medical domain classification
    medical_domains = {
        'medical_education': ['learning', 'education', 'training', 'teaching', 'competence', 'skill'],
        'clinical_practice': ['patient', 'clinical', 'treatment', 'diagnosis', 'therapy', 'care'],
        'public_health': ['population', 'epidemiology', 'prevention', 'health promotion', 'community'],
        'research_methodology': ['methodology', 'research design', 'study design', 'analysis', 'statistics'],
        'healthcare_policy': ['policy', 'healthcare system', 'administration', 'management', 'organization']
    }
    
    # Score document types
    type_scores = {}
    for doc_type, keywords in document_types.items():
        score = sum(clean_text.count(keyword) for keyword in keywords)
        type_scores[doc_type] = score
    
    # Score medical domains
    domain_scores = {}
    for domain, keywords in medical_domains.items():
        score = sum(clean_text.count(keyword) for keyword in keywords)
        domain_scores[domain] = score
    
    # Determine classifications
    document_type = max(type_scores, key=type_scores.get) if type_scores else 'unknown'
    medical_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'unknown'
    
    # Determine complexity level
    complexity_indicators = ['systematic', 'meta-analysis', 'statistical', 'methodology', 'quantitative']
    complexity_score = sum(clean_text.count(indicator) for indicator in complexity_indicators)
    
    if complexity_score >= 10:
        complexity_level = 'high'
    elif complexity_score >= 5:
        complexity_level = 'medium'
    else:
        complexity_level = 'low'
    
    classification = {
        'document_type': document_type,
        'medical_domain': medical_domain,
        'complexity_level': complexity_level,
        'confidence_score': f"{max(type_scores.values()) / sum(type_scores.values()) * 100:.1f}%" if sum(type_scores.values()) > 0 else "0%"
    }
    
    logger.info(f"Document classified as: {document_type} in {medical_domain} domain")
    return classification

def perform_comprehensive_nlp_analysis(text_content: str) -> MedicalNLPAnalysis:
    """
    Perform comprehensive NLP analysis on medical and scientific text.
    
    Args:
        text_content: The input text to analyze
        
    Returns:
        MedicalNLPAnalysis object containing all analysis results
    """
    logger.info("Starting comprehensive NLP analysis for medical/scientific text")
    
    if not text_content or len(text_content.strip()) < 100:
        logger.error("Insufficient text content for comprehensive NLP analysis")
        raise ValueError("Text content must be at least 100 characters long")
    
    try:
        # Extract text statistics
        text_stats = extract_text_statistics(text_content)
        
        # Extract terminology
        terminology = extract_medical_terminology(text_content)
        
        # Analyze semantic structure
        semantic_structure = analyze_semantic_structure(text_content)
        
        # Extract named entities
        named_entities = extract_named_entities(text_content)
        
        # Extract key phrases
        key_phrases = extract_key_phrases(text_content)
        
        # Calculate readability metrics
        readability = calculate_readability_metrics(text_content)
        
        # Classify document
        classification = classify_document_type(text_content)
        
        # Create comprehensive analysis result
        analysis_result = MedicalNLPAnalysis(
            text_statistics={
                'total_characters': text_stats.total_characters,
                'total_words': text_stats.total_words,
                'total_sentences': text_stats.total_sentences,
                'total_paragraphs': text_stats.total_paragraphs,
                'average_words_per_sentence': text_stats.average_words_per_sentence,
                'average_sentences_per_paragraph': text_stats.average_sentences_per_paragraph,
                'reading_level': text_stats.reading_level
            },
            terminology_extraction={
                'medical_terms': terminology.medical_terms,
                'scientific_terms': terminology.scientific_terms,
                'methodology_terms': terminology.methodology_terms,
                'statistical_terms': terminology.statistical_terms,
                'research_terms': terminology.research_terms
            },
            semantic_structure=semantic_structure,
            named_entities=named_entities,
            key_phrases=key_phrases,
            readability_metrics=readability,
            document_classification=classification
        )
        
        logger.info("✓ Comprehensive NLP analysis completed successfully")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error during comprehensive NLP analysis: {e}")
        raise

def save_nlp_analysis_results(analysis_result: MedicalNLPAnalysis, original_filename: str = "medical_text") -> Optional[str]:
    """
    Save the NLP analysis results to a JSON file.
    
    Args:
        analysis_result: The NLP analysis results to save
        original_filename: Base filename for the analysis file
        
    Returns:
        Path to saved analysis file, or None if saving fails
    """
    logger.info("Starting NLP analysis results save process")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate analysis filename
        base_name = Path(original_filename).stem
        analysis_filename = f"{base_name}_nlp_analysis.json"
        analysis_path = output_dir / analysis_filename
        
        # Convert to JSON-serializable format
        analysis_data = analysis_result.model_dump()
        
        # Write analysis to JSON file with pretty formatting
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ NLP analysis results saved successfully: {analysis_path}")
        return str(analysis_path)
        
    except Exception as e:
        logger.error(f"Error saving NLP analysis results: {e}")
        return None

# Helper functions

def _clean_markdown(text: str) -> str:
    """Remove markdown formatting for clean text analysis."""
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove markdown emphasis
    text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove page markers
    text = re.sub(r'<a id="page-\d+"></a>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _estimate_reading_level(avg_words_per_sentence: float) -> str:
    """Estimate reading level based on sentence complexity."""
    if avg_words_per_sentence < 15:
        return "Easy"
    elif avg_words_per_sentence < 20:
        return "Moderate"
    elif avg_words_per_sentence < 25:
        return "Difficult"
    else:
        return "Very Difficult"

def _identify_document_sections(text: str) -> List[str]:
    """Identify main sections in academic/medical documents."""
    sections = []
    section_patterns = [
        r'#+\s*(abstract|introduction|background|methodology|methods|results|discussion|conclusion|references|limitations|acknowledgments)',
        r'\b(abstract|introduction|background|methodology|methods|results|discussion|conclusion|references|limitations|acknowledgments)\b'
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sections.extend(matches)
    
    return list(set([section.lower() for section in sections]))

def _extract_research_methodology(text: str) -> Optional[str]:
    """Extract research methodology information from the text."""
    methodology_patterns = [
        r'(?:systematic review|meta-analysis|randomized controlled trial|qualitative study|quantitative study|mixed-methods|cross-sectional|longitudinal)',
        r'(?:survey|interview|focus group|observation|case study|literature review)'
    ]
    
    for pattern in methodology_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None

def _extract_study_population(text: str) -> Optional[str]:
    """Extract study population information from the text."""
    population_patterns = [
        r'\b(\d+)\s*(?:participants?|subjects?|patients?|individuals?|professionals?|nurses?|physicians?|doctors?)',
        r'(?:sample size|population|cohort|participants?).*?(\d+)',
        r'(\d+)\s*(?:healthcare professionals?|medical professionals?|clinicians?)'
    ]
    
    for pattern in population_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None

def _extract_key_findings(text: str) -> List[str]:
    """Extract key findings from the text."""
    findings = []
    finding_patterns = [
        r'(?:findings? revealed?|results? showed?|study found|analysis indicated?|research demonstrated?)([^.]+)',
        r'(?:significant|important|notable|key) (?:finding|result|outcome)([^.]+)',
        r'(?:the study|this research|our analysis) (?:shows?|indicates?|demonstrates?)([^.]+)'
    ]
    
    for pattern in finding_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        findings.extend([match.strip() for match in matches if len(match.strip()) > 20])
    
    return findings[:10]  # Limit to top 10 findings

def _extract_clinical_implications(text: str) -> List[str]:
    """Extract clinical implications from the text."""
    implications = []
    implication_patterns = [
        r'(?:clinical implications?|practical implications?|implications? for practice)([^.]+)',
        r'(?:these findings? suggest|this has implications)([^.]+)',
        r'(?:healthcare providers?|clinicians?) (?:should|must|need to)([^.]+)'
    ]
    
    for pattern in implication_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        implications.extend([match.strip() for match in matches if len(match.strip()) > 15])
    
    return implications[:5]  # Limit to top 5 implications

def _extract_limitations(text: str) -> List[str]:
    """Extract study limitations from the text."""
    limitations = []
    limitation_patterns = [
        r'(?:limitations?|constraints?|weaknesses?).*?include([^.]+)',
        r'(?:study limitations?|research limitations?)([^.]+)',
        r'(?:however|nevertheless|despite).*?limitation([^.]+)'
    ]
    
    for pattern in limitation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        limitations.extend([match.strip() for match in matches if len(match.strip()) > 10])
    
    return limitations[:5]  # Limit to top 5 limitations

def _is_relevant_phrase(phrase: str) -> bool:
    """Check if a phrase is relevant for medical/scientific context."""
    # Filter out common stop phrases and ensure minimum relevance
    stop_phrases = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    
    if any(phrase.startswith(stop) for stop in stop_phrases):
        return False
    
    # Check for medical/scientific relevance
    relevant_keywords = [
        'health', 'medical', 'clinical', 'patient', 'study', 'research', 'analysis',
        'treatment', 'therapy', 'diagnosis', 'learning', 'education', 'professional'
    ]
    
    return any(keyword in phrase for keyword in relevant_keywords)

def _count_syllables(word: str) -> int:
    """Count syllables in a word for readability calculations."""
    word = word.lower()
    syllables = 0
    vowels = 'aeiouy'
    
    if word[0] in vowels:
        syllables += 1
    
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            syllables += 1
    
    if word.endswith('e'):
        syllables -= 1
    
    if syllables == 0:
        syllables = 1
    
    return syllables

def _calculate_medical_complexity(text: str) -> float:
    """Calculate a custom medical complexity score based on terminology density."""
    # Count medical/scientific terms
    medical_terms = len(re.findall(r'\b(?:clinical|medical|therapeutic|diagnostic|pharmaceutical|pathological|physiological)\w*\b', text, re.IGNORECASE))
    technical_terms = len(re.findall(r'\b(?:methodology|statistical|systematic|quantitative|qualitative|empirical)\w*\b', text, re.IGNORECASE))
    
    # Count total words
    total_words = len(re.findall(r'\b\w+\b', text))
    
    if total_words == 0:
        return 0.0
    
    # Calculate complexity as percentage of specialized terms
    complexity = ((medical_terms + technical_terms) / total_words) * 100
    return min(100.0, complexity)

def test_nlp_analysis():
    """
    Test the NLP analysis functionality using the Self-directed learning journal article.
    
    This test demonstrates the complete NLP workflow for medical and scientific texts.
    """
    logger.info("=== Running NLP Analysis Tests ===")
    
    # Test 1: Check dependencies
    logger.info("Test 1: Checking dependencies and setup")
    try:
        import re
        from pathlib import Path
        from collections import Counter
        logger.info("✓ All dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False
    
    # Test 2: Load the journal article for testing
    logger.info("Test 2: Loading Self-directed learning journal article for testing")
    test_file_path = Path("output/Self-directed learning in health professions.md")
    
    if not test_file_path.exists():
        logger.error(f"✗ Test file not found: {test_file_path}")
        logger.info("Please ensure the journal article file is available for testing")
        return False
    
    content = load_text_content(str(test_file_path))
    if not content:
        logger.error("✗ Failed to load journal article content")
        return False
    
    logger.info(f"✓ Journal article loaded successfully ({len(content)} characters)")
    
    # Test 3: Test comprehensive NLP analysis
    logger.info("Test 3: Testing comprehensive NLP analysis")
    try:
        analysis_result = perform_comprehensive_nlp_analysis(content)
        logger.info("✓ Comprehensive NLP analysis completed successfully")
        
        # Display key results
        print(f"\n📊 NLP ANALYSIS RESULTS:")
        print(f"Document Type: {analysis_result.document_classification['document_type']}")
        print(f"Medical Domain: {analysis_result.document_classification['medical_domain']}")
        print(f"Total Words: {analysis_result.text_statistics['total_words']}")
        print(f"Reading Level: {analysis_result.text_statistics['reading_level']}")
        print(f"Medical Terms Found: {len(analysis_result.terminology_extraction['medical_terms'])}")
        print(f"Key Phrases: {len(analysis_result.key_phrases)}")
        
    except Exception as e:
        logger.error(f"✗ NLP analysis failed: {e}")
        return False
    
    # Test 4: Test saving analysis results
    logger.info("Test 4: Testing NLP analysis save functionality")
    saved_path = save_nlp_analysis_results(analysis_result, "Self-directed_learning_nlp_analysis")
    if not saved_path:
        logger.error("✗ Failed to save NLP analysis results")
        return False
    
    logger.info(f"✓ NLP analysis results saved successfully: {saved_path}")
    
    # Test 5: Verify saved analysis structure
    logger.info("Test 5: Verifying saved NLP analysis file structure")
    try:
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        logger.info(f"✓ NLP analysis file verified with {len(saved_data)} main sections")
        
        # Display analysis structure
        print(f"\n🔍 NLP ANALYSIS STRUCTURE:")
        print(f"   File: {saved_path}")
        print(f"   Sections: {list(saved_data.keys())}")
        print(f"   Medical Terms: {len(saved_data['terminology_extraction']['medical_terms'])}")
        print(f"   Named Entities: {sum(len(entities) for entities in saved_data['named_entities'].values())}")
        print(f"   Readability Score: {saved_data['readability_metrics'].get('flesch_reading_ease', 'N/A')}")
        
    except Exception as e:
        logger.error(f"✗ Error verifying NLP analysis file: {e}")
        return False
    
    logger.info("=== All NLP Analysis Tests Passed! ===")
    logger.info(f"Test completed using journal article: {test_file_path}")
    print(f"\n🎉 SUCCESS! The NLP analysis system successfully:")
    print(f"   ✓ Extracted comprehensive text statistics")
    print(f"   ✓ Identified medical and scientific terminology")
    print(f"   ✓ Analyzed semantic structure and document sections")
    print(f"   ✓ Extracted named entities and key phrases")
    print(f"   ✓ Calculated readability and complexity metrics")
    print(f"   ✓ Classified document type and medical domain")
    print(f"   ✓ Generated structured JSON output for further analysis")
    
    return True

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Run comprehensive tests with the Self-directed learning journal article
    test_nlp_analysis()
