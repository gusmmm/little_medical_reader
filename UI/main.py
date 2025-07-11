import streamlit as st
import logging
import os
import pdfplumber
from typing import Any
import sys
from pathlib import Path

# Add the parent directory to the Python path to import from file_processor and agents
sys.path.append(str(Path(__file__).parent.parent))
from file_processor.pdf_processor import process_pdf_to_session_state, get_markdown_from_session
from agents.summary_agent import process_medical_document, detect_medical_document_type, generate_adaptive_summary

# streamlit app configuration
st.set_page_config(
    page_title="Little Medical Reader",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# use pfdfplumber for extrcting the pdf pages as images
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
    
    # markdown content display and AI summary section
    with col2:
        # Get markdown content from session state
        markdown_content = get_markdown_from_session()
        
        if markdown_content:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📄 Original Content", "🤖 AI Summary"])
            
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
        
        else:
            st.info("Upload a PDF file to see the converted content and generate an AI summary.")

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

if __name__ == "__main__":
    main()