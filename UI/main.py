import streamlit as st
import logging
import os
import pdfplumber
from typing import Any
import sys
from pathlib import Path

# Configure logging BEFORE any other operations to ensure logger is available
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# streamlit app configuration
st.set_page_config(
    page_title="Little Medical Reader",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add the parent directory to the Python path to import from file_processor and agents
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules with error handling
try:
    from file_processor.pdf_processor import process_pdf_to_session_state, get_markdown_from_session
    logger.info("✓ PDF processor imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import PDF processor: {e}")
    st.error("Failed to import PDF processing modules. Please check your installation.")

try:
    from agents.summary_agent import detect_medical_document_type, generate_adaptive_summary
    logger.info("✓ Summary agent imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import summary agent: {e}")
    st.error("Failed to import summary agent modules. Please check your installation.")

try:
    from agents.concepts_agent import extract_medical_concepts
    logger.info("✓ Concepts agent imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import concepts agent: {e}")
    st.error("Failed to import concepts agent modules. Please check your installation.")

# use pdfplumber for extracting the pdf pages as images
@st.cache_data
def extract_all_pages_as_images(file_upload)-> list[Any]:
    """
    Extract all pages of a PDF file as images.
    
    Args:
        file_upload: The uploaded PDF file.
        
    Returns:
        A list of images extracted from the PDF.
    """
    logger.info(f"Extracting all pages as images from {file_upload.name}.")
    if not file_upload:
        logger.warning("No file uploaded.")
        return []

    images = []
    with pdfplumber.open(file_upload) as pdf:
        for page in pdf.pages:
            images.append(page.to_image().original)
        logger.info(f"Extracted {len(images)} pages as images from {file_upload.name}.")
    return images

def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    logger.info("Starting Little Medical Reader app")
    
    st.subheader("Welcome to Little Medical Reader", divider="grey",anchor=False)
    col1, col2 = st.columns([1.5, 2])

    # file upload section
    with col1:
        st.markdown("### Upload your medical file")
        file_upload = st.file_uploader(
            "Choose a file",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="file_uploader",
            help="Upload your medical document here."
        )
        if file_upload:
            st.success(f"File '{file_upload.name}' uploaded successfully!")
            
            # Process PDF to markdown and save to session state
            with st.spinner("Converting PDF to markdown..."):
                success, error_message = process_pdf_to_session_state(file_upload)
            
            if success:
                st.success("PDF converted to markdown successfully!")
            else:
                st.error(f"Failed to process PDF: {error_message}")
            
            # extract pages as images
            with st.spinner("Extracting pages..."):
                images = extract_all_pages_as_images(file_upload)
            if images:
                st.session_state["pdf_pages"] = images
                st.success(f"Extracted {len(images)} pages from the file.")
        
        # display extracted images
        if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
            st.markdown("### Extracted Pages")
            zoom_level = st.slider(
                "Zoom Level",
                min_value=100,
                max_value=1000,
                value=700,
                step=50,
                key="zoom_slider",
                help="Adjust the zoom level for the images."
            )
            with st.container(height=600, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(
                        page_image,
                        caption="Extracted Page",
                        width=zoom_level,
                    )
    
    # markdown content display and AI analysis section
    with col2:
        # Get markdown content from session state
        markdown_content = get_markdown_from_session()
        
        if markdown_content:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📄 Original Content", "🤖 AI Summary", "🧠 Medical Concepts"])
            
            with tab1:
                st.markdown("### Converted Markdown Content")
                # Show markdown content in an expandable container
                with st.container(height=500, border=True):
                    st.markdown(markdown_content)
                
                # Add download button for the markdown content
                st.download_button(
                    label="Download Markdown File",
                    data=markdown_content,
                    file_name=f"{st.session_state.get('original_pdf_name', 'document').replace('.pdf', '.md')}",
                    mime="text/markdown",
                    help="Download the converted markdown content"
                )
            
            with tab2:
                st.markdown("### AI-Generated Medical Summary")
                
                # Add generate summary button
                if st.button("🚀 Generate AI Summary", type="primary", help="Generate an intelligent summary using AI"):
                    generate_ai_summary(markdown_content)
                
                # Display existing summary if available
                display_ai_summary()
            
            with tab3:
                st.markdown("### Medical Concepts Extraction")
                
                # Add generate concepts button
                if st.button("🧠 Extract Medical Concepts", type="primary", help="Extract medical concepts and relationships using AI"):
                    generate_medical_concepts(markdown_content)
                
                # Display existing concepts if available
                display_medical_concepts()
        
        else:
            st.info("Upload a PDF file to see the converted content and generate AI analysis.")

def generate_ai_summary(content: str) -> None:
    """
    Generate AI summary from markdown content and store in session state.
    
    Args:
        content: The markdown content to summarize
    """
    logger.info("Starting AI summary generation process")
    
    try:
        with st.spinner("🔍 Analyzing document type..."):
            # Step 1: Detect document type
            analysis = detect_medical_document_type(content)
            
            if "error" in analysis:
                st.error(f"Failed to analyze document: {analysis['error']}")
                logger.error(f"Document analysis failed: {analysis['error']}")
                return
            
            # Store analysis in session state
            st.session_state["document_analysis"] = analysis
            
            # Display analysis results
            with st.expander("📊 Document Analysis Results", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Document Type", analysis.get('document_type', 'Unknown'))
                    st.metric("Medical Specialty", analysis.get('medical_specialty', 'Unknown'))
                with col2:
                    st.metric("Complexity Level", analysis.get('complexity_level', 'Unknown'))
                    st.metric("Target Audience", analysis.get('target_audience', 'Unknown'))
        
        with st.spinner("🤖 Generating intelligent summary..."):
            # Step 2: Generate adaptive summary
            summary = generate_adaptive_summary(content, analysis)
            
            if not summary:
                st.error("Failed to generate AI summary. Please try again.")
                logger.error("Summary generation failed")
                return
            
            # Store summary in session state
            st.session_state["ai_summary"] = summary
            st.session_state["summary_generated"] = True
            
            st.success("✅ AI summary generated successfully!")
            logger.info("AI summary generated and stored in session state")
            
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {str(e)}")
        logger.error(f"Error in generate_ai_summary: {str(e)}")

def display_ai_summary() -> None:
    """
    Display the AI-generated summary if available in session state.
    """
    if st.session_state.get("summary_generated", False) and "ai_summary" in st.session_state:
        summary = st.session_state["ai_summary"]
        analysis = st.session_state.get("document_analysis", {})
        
        # Display summary metadata
        if analysis:
            with st.expander("📋 Summary Information", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Document Type:** {analysis.get('document_type', 'Unknown')}")
                with col2:
                    st.write(f"**Medical Specialty:** {analysis.get('medical_specialty', 'Unknown')}")
                with col3:
                    st.write(f"**Summary Strategy:** {analysis.get('summary_strategy', 'Unknown')}")
        
        # Display the summary content
        with st.container(height=500, border=True):
            st.markdown(summary)
        
        # Add download button for the summary
        original_filename = st.session_state.get('original_pdf_name', 'document')
        summary_filename = f"{Path(original_filename).stem}_AI_summary.md"
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download AI Summary",
                data=summary,
                file_name=summary_filename,
                mime="text/markdown",
                help="Download the AI-generated summary as markdown"
            )
        
        with col2:
            if st.button("🔄 Regenerate Summary", help="Generate a new AI summary"):
                # Clear existing summary to trigger regeneration
                if "ai_summary" in st.session_state:
                    del st.session_state["ai_summary"]
                if "summary_generated" in st.session_state:
                    del st.session_state["summary_generated"]
                st.rerun()
    
    elif "markdown_content" in st.session_state:
        # Show placeholder when content is available but no summary generated yet
        with st.container(height=200, border=True):
            st.info("🤖 Click 'Generate AI Summary' above to create an intelligent summary of your medical document.")
            st.write("The AI will:")
            st.write("• 🔍 Analyze the document type and medical specialty")
            st.write("• 📝 Generate a structured summary appropriate for healthcare professionals") 
            st.write("• 🎯 Focus on key medical findings, treatments, and recommendations")

def generate_medical_concepts(content: str) -> None:
    """
    Generate medical concepts extraction from markdown content and store in session state.
    Includes progress tracking and persistent processing to prevent timeouts.
    
    Args:
        content: The markdown content to analyze for medical concepts
    """
    logger.info("Starting medical concepts extraction process with progress tracking")
    
    try:
        # Initialize progress tracking in session state
        st.session_state["concepts_processing"] = True
        st.session_state["concepts_progress"] = 0
        st.session_state["concepts_status"] = "Initializing concept extraction..."
        st.session_state["concepts_total_chunks"] = 0
        st.session_state["concepts_current_chunk"] = 0
        
        # Create progress containers
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Starting medical concepts extraction process...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            chunk_info = st.empty()
        
        # Update status
        status_text.text("🔍 Analyzing text and preparing for concept extraction...")
        
        # Try to import and use the progress function, fall back to regular function if not available
        try:
            from agents.concepts_agent import extract_medical_concepts_with_progress
            logger.info("Using advanced progress tracking for concept extraction")
            
            # Start extraction with progress callback
            def update_progress(current_chunk: int, total_chunks: int, status: str):
                """Callback function to update progress in the UI"""
                if total_chunks > 0:
                    progress = current_chunk / total_chunks
                    st.session_state["concepts_progress"] = progress
                    st.session_state["concepts_current_chunk"] = current_chunk
                    st.session_state["concepts_total_chunks"] = total_chunks
                    st.session_state["concepts_status"] = status
                    
                    # Update UI elements
                    progress_bar.progress(progress)
                    status_text.text(f"🧠 {status}")
                    chunk_info.text(f"Processing chunk {current_chunk} of {total_chunks}")
                    
                    logger.info(f"Progress update: {current_chunk}/{total_chunks} - {status}")
            
            # Extract medical concepts with progress tracking
            extraction_result = extract_medical_concepts_with_progress(content, progress_callback=update_progress)
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Progress function not available ({e}), using standard extraction")
            # Fall back to regular extraction without progress tracking
            
            progress_bar.progress(0.5)
            status_text.text("🧠 Extracting medical concepts...")
            chunk_info.text("Processing medical concepts extraction...")
            
            extraction_result = extract_medical_concepts(content)
        
        # Clear processing flags
        st.session_state["concepts_processing"] = False
        
        if not extraction_result:
            st.error("Failed to extract medical concepts. Please try again.")
            logger.error("Medical concepts extraction failed")
            return
        
        # Store extraction results in session state
        st.session_state["medical_concepts"] = extraction_result
        st.session_state["concepts_generated"] = True
        
        # Update final progress
        progress_bar.progress(1.0)
        status_text.text("✅ Medical concepts extraction completed successfully!")
        chunk_info.text(f"Extracted {extraction_result.total_concepts} medical concepts")
        
        # Display extraction statistics
        with st.expander("📊 Extraction Statistics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Concepts", extraction_result.total_concepts)
            with col2:
                total_relationships = len([c for c in extraction_result.concepts if c.related_concept_id])
                st.metric("Total Relationships", total_relationships)
            with col3:
                concept_types = list(set(c.concept_type for c in extraction_result.concepts))
                st.metric("Concept Types", len(concept_types))
        
        st.success("✅ Medical concepts extracted successfully!")
        logger.info("Medical concepts extracted and stored in session state")
        
        # Auto-refresh the display after a short delay
        st.rerun()
            
    except Exception as e:
        # Clear processing flags on error
        st.session_state["concepts_processing"] = False
        st.error(f"An error occurred while extracting medical concepts: {str(e)}")
        logger.error(f"Error in generate_medical_concepts: {str(e)}")

def display_medical_concepts() -> None:
    """
    Display the extracted medical concepts if available in session state.
    Also handles progress display during ongoing processing.
    """
    # Check if processing is in progress
    if st.session_state.get("concepts_processing", False):
        st.markdown("#### 🔄 Medical Concepts Extraction in Progress")
        
        # Show persistent progress indicators
        progress = st.session_state.get("concepts_progress", 0)
        current_chunk = st.session_state.get("concepts_current_chunk", 0)
        total_chunks = st.session_state.get("concepts_total_chunks", 0)
        status = st.session_state.get("concepts_status", "Processing...")
        
        # Progress bar
        progress_bar = st.progress(progress)
        
        # Status information
        if total_chunks > 0:
            st.info(f"🧠 {status}")
            st.text(f"Processing chunk {current_chunk} of {total_chunks}")
        else:
            st.info("🔍 Initializing medical concept extraction...")
        
        # Warning about not changing tabs
        st.warning("⚠️ Processing in progress. You can safely navigate to other tabs - the extraction will continue in the background.")
        
        # Auto-refresh every 2 seconds during processing
        import time
        time.sleep(2)
        st.rerun()
        
        return
    
    if st.session_state.get("concepts_generated", False) and "medical_concepts" in st.session_state:
        extraction_result = st.session_state["medical_concepts"]
        
        # Display concepts overview
        st.markdown("#### 🏥 Medical Concepts Overview")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Concepts", extraction_result.total_concepts)
        with col2:
            total_relationships = len([c for c in extraction_result.concepts if c.related_concept_id])
            st.metric("Relationships", total_relationships)
        with col3:
            concept_types = list(set(c.concept_type for c in extraction_result.concepts))
            st.metric("Types Found", len(concept_types))
        with col4:
            specialties = list(set(c.medical_specialty for c in extraction_result.concepts if c.medical_specialty))
            st.metric("Specialties", len(specialties))
        
        # Show filtering information
        if extraction_result.total_concepts == 50:
            st.info("📊 Results filtered to show the 50 most relevant concepts based on frequency and clinical significance.")
        
        # Display concept type distribution
        with st.expander("📋 Concept Type Distribution", expanded=False):
            concept_type_counts = {}
            for concept in extraction_result.concepts:
                concept_type_counts[concept.concept_type] = concept_type_counts.get(concept.concept_type, 0) + 1
            
            for concept_type, count in concept_type_counts.items():
                st.write(f"**{concept_type.replace('_', ' ').title()}**: {count}")
        
        # Display concept category distribution
        with st.expander("🏷️ Medical Category Distribution", expanded=False):
            category_counts = {}
            for concept in extraction_result.concepts:
                category_counts[concept.concept_category] = category_counts.get(concept.concept_category, 0) + 1
            
            for category, count in category_counts.items():
                st.write(f"**{category.replace('_', ' ').title()}**: {count}")
        
        # Display sample concepts
        st.markdown("#### 🔬 Sample Medical Concepts")
        
        # Show first 10 concepts in an expandable container
        with st.container(height=400, border=True):
            for i, concept in enumerate(extraction_result.concepts[:10]):
                with st.expander(f"{i+1}. {concept.concept_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**ID**: {concept.concept_id}")
                        st.write(f"**Type**: {concept.concept_type}")
                        st.write(f"**Category**: {concept.concept_category}")
                        st.write(f"**Specialty**: {concept.medical_specialty or 'General Medicine'}")
                    with col2:
                        st.write(f"**Clinical Significance**: {concept.clinical_significance or 'Not specified'}")
                        st.write(f"**Confidence**: {concept.confidence_score:.2f}")
                        if concept.related_concept_name:
                            st.write(f"**Relationship**: {concept.relationship_type} → {concept.related_concept_name}")
                            st.write(f"**Strength**: {concept.relationship_strength:.2f}")
                    
                    st.write(f"**Context**: {concept.text_context[:150]}...")
        
        if len(extraction_result.concepts) > 10:
            st.info(f"Showing first 10 concepts. Total of {len(extraction_result.concepts)} concepts extracted.")
        
        # Display relationship types
        with st.expander("🔗 Clinical Relationship Types", expanded=False):
            relationship_counts = {}
            for concept in extraction_result.concepts:
                rel_type = concept.relationship_type
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
            
            for rel_type, count in sorted(relationship_counts.items()):
                st.write(f"**{rel_type}**: {count}")
        
        # Display medical specialties
        specialties = set(c.medical_specialty for c in extraction_result.concepts if c.medical_specialty)
        if specialties:
            with st.expander("🏥 Medical Specialties Identified", expanded=False):
                specialty_counts = {}
                for concept in extraction_result.concepts:
                    if concept.medical_specialty:
                        specialty_counts[concept.medical_specialty] = specialty_counts.get(concept.medical_specialty, 0) + 1
                
                for specialty, count in sorted(specialty_counts.items()):
                    st.write(f"**{specialty.replace('_', ' ').title()}**: {count}")
        
        # Add download functionality
        st.markdown("#### 📥 Download Options")
        
        col1, col2 = st.columns(2)
        with col1:
            # Create JSON data for download
            import json
            enhanced_data = {
                "extraction_metadata": {
                    "source_file": st.session_state.get('original_pdf_name', 'document'),
                    "total_concepts": extraction_result.total_concepts,
                    "total_relationships": len([c for c in extraction_result.concepts if c.related_concept_id]),
                    "extraction_model": "gemini-2.0-flash",
                    "filtering_applied": "top_50_most_relevant",
                    "concept_types_found": list(set(c.concept_type for c in extraction_result.concepts)),
                    "concept_categories_found": list(set(c.concept_category for c in extraction_result.concepts)),
                    "relationship_types_found": list(set(c.relationship_type for c in extraction_result.concepts)),
                    "medical_specialties_found": list(set(c.medical_specialty for c in extraction_result.concepts if c.medical_specialty))
                },
                "medical_concepts": [concept.model_dump() for concept in extraction_result.concepts],
                "total_concepts": extraction_result.total_concepts
            }
            
            json_data = json.dumps(enhanced_data, indent=2, ensure_ascii=False)
            original_filename = st.session_state.get('original_pdf_name', 'document')
            concepts_filename = f"{Path(original_filename).stem}_medical_concepts.json"
            
            st.download_button(
                label="📥 Download Medical Concepts (JSON)",
                data=json_data,
                file_name=concepts_filename,
                mime="application/json",
                help="Download the extracted medical concepts as JSON for Neo4j import"
            )
        
        with col2:
            if st.button("🔄 Regenerate Concepts", help="Extract medical concepts again"):
                # Clear existing concepts to trigger regeneration
                if "medical_concepts" in st.session_state:
                    del st.session_state["medical_concepts"]
                if "concepts_generated" in st.session_state:
                    del st.session_state["concepts_generated"]
                # Clear progress tracking variables
                for key in ["concepts_processing", "concepts_progress", "concepts_status", 
                           "concepts_total_chunks", "concepts_current_chunk"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    elif "markdown_content" in st.session_state:
        # Show placeholder when content is available but no concepts generated yet
        with st.container(height=300, border=True):
            st.info("🧠 Click 'Extract Medical Concepts' above to analyze your medical document for key concepts and relationships.")
            st.write("The AI will:")
            st.write("• 🔍 Identify medical concepts (diseases, medications, symptoms, procedures)")
            st.write("• 📊 Categorize concepts by type (individual, qualitative, comparative, quantitative)")
            st.write("• 🔗 Extract relationships between medical concepts")
            st.write("• 🏥 Identify relevant medical specialties")
            st.write("• 📄 Generate Neo4j-ready JSON for knowledge graphs")
            st.write("• 🎯 Filter to the 50 most relevant concepts")
            
            st.markdown("**Medical Concept Types:**")
            st.write("• **Individual**: Specific entities (e.g., 'acute myocardial infarction', 'metformin 500mg')")
            st.write("• **Qualitative**: Classifications (e.g., 'hypertension', 'positive blood culture')")
            st.write("• **Comparative**: Comparisons (e.g., 'worse than yesterday', 'above normal range')")
            st.write("• **Quantitative**: Measurements (e.g., 'blood pressure 140/90 mmHg', 'hemoglobin 8.2 g/dL')")

if __name__ == "__main__":
    main()