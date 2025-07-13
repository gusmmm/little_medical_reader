import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from google import genai
from dotenv import load_dotenv
import json
import re
from pydantic import BaseModel, Field
from datetime import datetime

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

class CitationInfo(BaseModel):
    """Citation information for the medical article"""
    title: str = Field(description="Full title of the article")
    doi: Optional[str] = Field(description="DOI of the article if available")
    publication_date: Optional[str] = Field(description="Publication date")
    first_author: Optional[str] = Field(description="First author name")
    second_author: Optional[str] = Field(description="Second author name if available")
    journal_name: Optional[str] = Field(description="Name of the journal")
    url_link: Optional[str] = Field(description="URL link to the article if present")

class ArticleAnalysis(BaseModel):
    """Comprehensive analysis of a medical journal article"""
    citation_info: CitationInfo = Field(description="Citation and publication information")
    abstract_keywords: List[str] = Field(description="Key terms and keywords from the abstract")
    abstract_summary: str = Field(description="Concise summary of the abstract")
    methodology: str = Field(description="Description of the study methodology and approach")
    main_results: str = Field(description="Summary of the main findings and results")
    discussion_points: Dict[str, List[str]] = Field(
        description="Discussion analysis with good_qualities and bad_qualities as keys"
    )
    main_conclusions: str = Field(description="Primary conclusions drawn from the study")
    questions_raised: List[str] = Field(description="Questions or uncertainties raised by the results")
    curiosities: List[str] = Field(description="Interesting points worth deeper investigation")

def find_enhanced_markdown_file(output_folder: str) -> Optional[str]:
    """
    Find the enhanced markdown file in the specified output folder.
    
    Args:
        output_folder: Path to the output folder (e.g., 'output/jcm-12-03188')
        
    Returns:
        Path to the enhanced markdown file, or None if not found
    """
    logger.info(f"Searching for enhanced markdown file in: {output_folder}")
    
    try:
        folder_path = Path(output_folder)
        if not folder_path.exists():
            logger.error(f"Output folder not found: {output_folder}")
            return None
        
        # Look for files ending with '_enhanced.md'
        enhanced_files = list(folder_path.glob("*_enhanced.md"))
        
        if enhanced_files:
            enhanced_file = enhanced_files[0]  # Take the first enhanced file found
            logger.info(f"✓ Found enhanced markdown file: {enhanced_file}")
            return str(enhanced_file)
        else:
            # Fallback: look for any .md file that might be the processed document
            md_files = list(folder_path.glob("*.md"))
            if md_files:
                # Prefer files that don't end with '_summary.md' or '_concepts.md'
                main_files = [f for f in md_files if not any(suffix in f.name for suffix in ['_summary', '_concepts'])]
                if main_files:
                    logger.info(f"✓ Found main markdown file: {main_files[0]}")
                    return str(main_files[0])
                else:
                    logger.info(f"✓ Using available markdown file: {md_files[0]}")
                    return str(md_files[0])
            else:
                logger.error("No markdown files found in the output folder")
                return None
                
    except Exception as e:
        logger.error(f"Error searching for enhanced markdown file: {e}")
        return None

def load_enhanced_markdown_content(output_folder: str) -> Optional[str]:
    """
    Load the enhanced markdown content from the specified output folder.
    
    Args:
        output_folder: Path to the output folder containing processed files
        
    Returns:
        Content of the enhanced markdown file, or None if loading fails
    """
    logger.info(f"Loading enhanced markdown content from: {output_folder}")
    
    enhanced_file_path = find_enhanced_markdown_file(output_folder)
    if not enhanced_file_path:
        return None
    
    try:
        with open(enhanced_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"✓ Successfully loaded enhanced markdown content")
        logger.info(f"Content length: {len(content)} characters")
        return content
        
    except Exception as e:
        logger.error(f"Error reading enhanced markdown file {enhanced_file_path}: {e}")
        return None

def analyze_medical_article(content: str) -> Optional[ArticleAnalysis]:
    """
    Analyze a medical journal article using Gemini AI and return structured data.
    
    Args:
        content: The enhanced markdown content to analyze
        
    Returns:
        ArticleAnalysis object with structured analysis, or None if analysis fails
    """
    logger.info("Starting comprehensive medical article analysis")
    
    if not client:
        logger.error("Gemini client not available for article analysis")
        return None
    
    analysis_prompt = f"""
    You are an expert medical researcher and analyst. Analyze the following medical journal article and extract comprehensive information.

    ARTICLE CONTENT:
    {content}

    Please analyze this article and respond with ONLY a valid JSON object that matches this exact structure:
    {{
        "citation_info": {{
            "title": "Full title of the article",
            "doi": "DOI if available or null",
            "publication_date": "Publication date if available or null",
            "first_author": "First author name or null",
            "second_author": "Second author name if available or null", 
            "journal_name": "Journal name if available or null",
            "url_link": "URL link if present or null"
        }},
        "abstract_keywords": ["keyword1", "keyword2", "keyword3"],
        "abstract_summary": "Concise 2-3 sentence summary of the abstract",
        "methodology": "Detailed description of the study methodology, design, participants, and approach",
        "main_results": "Summary of the key findings, statistical results, and primary outcomes",
        "discussion_points": {{
            "good_qualities": ["strength1", "strength2", "strength3"],
            "bad_qualities": ["limitation1", "limitation2", "limitation3"]
        }},
        "main_conclusions": "Primary conclusions and clinical implications drawn from the study",
        "questions_raised": ["question1", "question2", "question3"],
        "curiosities": ["interesting_point1", "interesting_point2", "interesting_point3"]
    }}

    IMPORTANT GUIDELINES:
    - Extract information exactly as it appears in the article
    - For missing information, use null for strings and empty arrays for lists
    - Focus on medical and clinical relevance
    - Be thorough in identifying methodology details
    - Include statistical significance in results when available
    - Identify both strengths and limitations objectively
    - Generate thoughtful questions based on the research gaps or findings
    - Highlight unique or novel aspects as curiosities

    Return ONLY valid JSON, no other text.
    """
    
    try:
        logger.info("Sending article for AI analysis")
        
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
        
        # Parse JSON and validate with Pydantic
        analysis_data = json.loads(json_text)
        analysis_result = ArticleAnalysis(**analysis_data)
        
        logger.info("✓ Medical article analysis completed successfully")
        logger.info(f"Article title: {analysis_result.citation_info.title[:50]}...")
        logger.info(f"Keywords extracted: {len(analysis_result.abstract_keywords)}")
        logger.info(f"Questions raised: {len(analysis_result.questions_raised)}")
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in article analysis: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in medical article analysis: {e}")
        return None

def save_analysis_results(analysis: ArticleAnalysis, output_folder: str) -> Optional[str]:
    """
    Save the structured analysis results to JSON file in the output folder.
    
    Args:
        analysis: ArticleAnalysis object containing the structured data
        output_folder: Path to the output folder
        
    Returns:
        Path to saved analysis file, or None if saving fails
    """
    logger.info("Saving analysis results to file")
    
    try:
        folder_path = Path(output_folder)
        folder_name = folder_path.name
        
        # Create filename for the analysis
        analysis_filename = f"{folder_name}_article_analysis.json"
        analysis_path = folder_path / analysis_filename
        
        # Convert Pydantic model to dict and add timestamp
        analysis_dict = analysis.model_dump()
        analysis_dict["timestamp_created"] = datetime.now().isoformat()
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Analysis results saved successfully: {analysis_path}")
        return str(analysis_path)
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")
        return None

def process_medical_article_v2(output_folder: str) -> Optional[ArticleAnalysis]:
    """
    Complete V2 workflow to process medical article from enhanced markdown.
    
    Args:
        output_folder: Path to the output folder containing enhanced markdown
        
    Returns:
        ArticleAnalysis object with structured results, or None if processing fails
    """
    logger.info("=== Starting V2 Medical Article Processing Workflow ===")
    
    try:
        # Step 1: Load enhanced markdown content
        logger.info("Step 1: Loading enhanced markdown content")
        content = load_enhanced_markdown_content(output_folder)
        if not content:
            logger.error("Failed to load enhanced markdown content")
            return None
        
        # Step 2: Analyze article with structured extraction
        logger.info("Step 2: Analyzing article with structured extraction")
        analysis = analyze_medical_article(content)
        if not analysis:
            logger.error("Failed to analyze medical article")
            return None
        
        # Step 3: Save structured analysis results
        logger.info("Step 3: Saving structured analysis results")
        analysis_path = save_analysis_results(analysis, output_folder)
        if not analysis_path:
            logger.error("Failed to save analysis results")
            return None
        
        logger.info("=== V2 Medical Article Processing Completed Successfully ===")
        return analysis
        
    except Exception as e:
        logger.error(f"Unexpected error in V2 medical article processing: {e}")
        return None

def test_v2_summary_agent():
    """
    Test function for the V2 Summary Agent using the jcm-12-03188 folder.
    Tests the complete workflow with real processed medical article data.
    """
    logger.info("=== Running V2 Summary Agent Tests ===")
    
    # Test configuration
    test_output_folder = "output/jcm-12-03188"
    
    # Test 1: Check dependencies and setup
    logger.info("Test 1: Checking dependencies and setup")
    try:
        from google import genai
        from dotenv import load_dotenv
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
    
    # Test 3: Check test folder exists
    logger.info("Test 3: Verifying test folder exists")
    test_folder_path = Path(test_output_folder)
    if not test_folder_path.exists():
        logger.error(f"✗ Test folder not found: {test_output_folder}")
        logger.info("Please ensure the jcm-12-03188 folder exists in the output directory")
        return False
    
    logger.info(f"✓ Test folder found: {test_output_folder}")
    
    # Test 4: Find enhanced markdown file
    logger.info("Test 4: Testing enhanced markdown file detection")
    enhanced_file = find_enhanced_markdown_file(test_output_folder)
    if not enhanced_file:
        logger.error("✗ Failed to find enhanced markdown file")
        return False
    
    logger.info(f"✓ Enhanced markdown file found: {enhanced_file}")
    
    # Test 5: Load enhanced markdown content
    logger.info("Test 5: Testing enhanced markdown content loading")
    content = load_enhanced_markdown_content(test_output_folder)
    if not content:
        logger.error("✗ Failed to load enhanced markdown content")
        return False
    
    logger.info(f"✓ Enhanced markdown content loaded successfully")
    logger.info(f"Content length: {len(content)} characters")
    
    # Test 6: Analyze medical article with structured extraction
    logger.info("Test 6: Testing structured medical article analysis")
    analysis = analyze_medical_article(content)
    if not analysis:
        logger.error("✗ Failed to analyze medical article")
        return False
    
    logger.info("✓ Medical article analysis completed successfully")
    logger.info(f"Article title: {analysis.citation_info.title[:80]}...")
    logger.info(f"DOI: {analysis.citation_info.doi}")
    logger.info(f"First author: {analysis.citation_info.first_author}")
    logger.info(f"Keywords found: {len(analysis.abstract_keywords)}")
    logger.info(f"Questions raised: {len(analysis.questions_raised)}")
    logger.info(f"Curiosities identified: {len(analysis.curiosities)}")
    
    # Test 7: Test Pydantic model validation
    logger.info("Test 7: Testing Pydantic model validation")
    try:
        # Verify all required fields are present
        assert analysis.citation_info.title, "Title should not be empty"
        assert analysis.abstract_summary, "Abstract summary should not be empty"
        assert analysis.methodology, "Methodology should not be empty"
        assert analysis.main_results, "Main results should not be empty"
        assert analysis.main_conclusions, "Main conclusions should not be empty"
        assert isinstance(analysis.abstract_keywords, list), "Keywords should be a list"
        assert isinstance(analysis.questions_raised, list), "Questions should be a list"
        assert isinstance(analysis.curiosities, list), "Curiosities should be a list"
        
        logger.info("✓ Pydantic model validation passed")
    except AssertionError as e:
        logger.error(f"✗ Pydantic model validation failed: {e}")
        return False
    
    # Test 8: Save analysis results
    logger.info("Test 8: Testing analysis results saving")
    analysis_path = save_analysis_results(analysis, test_output_folder)
    if not analysis_path:
        logger.error("✗ Failed to save analysis results")
        return False
    
    logger.info(f"✓ Analysis results saved successfully: {analysis_path}")
    
    # Test 9: Verify saved analysis file
    logger.info("Test 9: Verifying saved analysis file")
    if Path(analysis_path).exists():
        with open(analysis_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Verify the saved data structure
        required_keys = ['citation_info', 'abstract_keywords', 'abstract_summary', 
                        'methodology', 'main_results', 'discussion_points', 
                        'main_conclusions', 'questions_raised', 'curiosities']
        
        for key in required_keys:
            if key not in saved_data:
                logger.error(f"✗ Missing key in saved data: {key}")
                return False
        
        logger.info("✓ Saved analysis file verified successfully")
        logger.info(f"Saved data contains {len(saved_data)} top-level fields")
    else:
        logger.error("✗ Analysis file not found after saving")
        return False
    
    # Test 10: Complete workflow test
    logger.info("Test 10: Testing complete V2 processing workflow")
    workflow_analysis = process_medical_article_v2(test_output_folder)
    
    if workflow_analysis:
        logger.info("✓ Complete V2 workflow test passed")
        
        # Display detailed analysis results
        logger.info("=== DETAILED ANALYSIS RESULTS ===")
        logger.info(f"Title: {workflow_analysis.citation_info.title}")
        logger.info(f"DOI: {workflow_analysis.citation_info.doi}")
        logger.info(f"Authors: {workflow_analysis.citation_info.first_author}, {workflow_analysis.citation_info.second_author}")
        logger.info(f"Journal: {workflow_analysis.citation_info.journal_name}")
        logger.info(f"Publication Date: {workflow_analysis.citation_info.publication_date}")
        
        logger.info(f"\nAbstract Keywords ({len(workflow_analysis.abstract_keywords)}):")
        for i, keyword in enumerate(workflow_analysis.abstract_keywords[:5], 1):
            logger.info(f"  {i}. {keyword}")
        
        logger.info(f"\nAbstract Summary:")
        logger.info(f"  {workflow_analysis.abstract_summary}")
        
        logger.info(f"\nMethodology Summary:")
        logger.info(f"  {workflow_analysis.methodology[:200]}...")
        
        logger.info(f"\nMain Results Summary:")
        logger.info(f"  {workflow_analysis.main_results[:200]}...")
        
        logger.info(f"\nDiscussion Points:")
        logger.info(f"  Good Qualities: {len(workflow_analysis.discussion_points.get('good_qualities', []))}")
        logger.info(f"  Bad Qualities: {len(workflow_analysis.discussion_points.get('bad_qualities', []))}")
        
        logger.info(f"\nMain Conclusions:")
        logger.info(f"  {workflow_analysis.main_conclusions[:200]}...")
        
        logger.info(f"\nQuestions Raised ({len(workflow_analysis.questions_raised)}):")
        for i, question in enumerate(workflow_analysis.questions_raised[:3], 1):
            logger.info(f"  {i}. {question}")
        
        logger.info(f"\nCuriosities ({len(workflow_analysis.curiosities)}):")
        for i, curiosity in enumerate(workflow_analysis.curiosities[:3], 1):
            logger.info(f"  {i}. {curiosity}")
        
        logger.info("=== END DETAILED RESULTS ===")
    else:
        logger.error("✗ Complete V2 workflow test failed")
        return False
    
    logger.info("=== All V2 Summary Agent Tests Passed! ===")
    logger.info(f"Test completed using folder: {test_output_folder}")
    logger.info("The V2 system successfully:")
    logger.info("  - Found and loaded enhanced markdown file")
    logger.info("  - Extracted structured citation information")
    logger.info("  - Identified abstract keywords and summary")
    logger.info("  - Analyzed methodology and results")
    logger.info("  - Evaluated discussion points (strengths/weaknesses)")
    logger.info("  - Captured main conclusions")
    logger.info("  - Generated relevant research questions")
    logger.info("  - Identified interesting curiosities for further investigation")
    logger.info("  - Saved all results in structured JSON format")
    
    return True

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Run comprehensive tests with the jcm-12-03188 folder
    test_v2_summary_agent()