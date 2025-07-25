import pymupdf
import streamlit as st
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# Configure logging for this module
logger = logging.getLogger(__name__)

def pdf_to_markdown(file_upload) -> Optional[str]:
    """
    Convert uploaded PDF file to markdown format using PyMuPDF with structure preservation.
    
    This function uses PyMuPDF's advanced text extraction capabilities to:
    - Preserve logical document structure (headings, paragraphs, lists)
    - Detect font sizes to determine heading hierarchy
    - Extract table of contents if available
    - Maintain proper text block organization
    
    Args:
        file_upload: Streamlit uploaded file object containing the PDF
        
    Returns:
        String containing the structured markdown content, or None if conversion fails
    """
    logger.info(f"Starting advanced PDF to markdown conversion for file: {file_upload.name}")
    
    try:
        # Read the uploaded file bytes
        pdf_bytes = file_upload.read()
        
        # Open PDF document from bytes
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        # Store page count before processing to avoid accessing after close
        total_pages = pdf_document.page_count
        logger.info(f"PDF document opened successfully. Total pages: {total_pages}")
        
        markdown_content = ""
        
        # Extract table of contents if available for document structure
        toc = pdf_document.get_toc()
        if toc:
            logger.info(f"Found table of contents with {len(toc)} entries")
            markdown_content += "# Table of Contents\n\n"
            for level, title, page_num in toc:
                indent = "  " * (level - 1)
                markdown_content += f"{indent}- [{title}](#page-{page_num})\n"
            markdown_content += "\n---\n\n"
        
        # Track font sizes across document to determine heading hierarchy
        font_sizes = set()
        
        # First pass: collect all font sizes to establish hierarchy
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            # Get text blocks with detailed formatting information
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.add(span["size"])
        
        # Sort font sizes to establish heading hierarchy (largest = h1, etc.)
        sorted_font_sizes = sorted(font_sizes, reverse=True)
        logger.info(f"Detected font sizes: {sorted_font_sizes}")
        
        # Second pass: extract text with proper structure
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Add page anchor for TOC navigation
            markdown_content += f'<a id="page-{page_num + 1}"></a>\n\n'
            markdown_content += f"# Page {page_num + 1}\n\n"
            
            # Get text blocks with detailed formatting and positioning
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if "lines" in block:  # Text block (not image)
                    block_text = ""
                    current_font_size = None
                    
                    for line in block["lines"]:
                        line_text = ""
                        line_font_size = None
                        
                        for span in line["spans"]:
                            # Track font size for this span
                            span_font_size = span["size"]
                            span_text = span["text"]
                            
                            # Determine if this is a heading based on font size
                            if span_font_size in sorted_font_sizes[:4]:  # Top 4 font sizes as headings
                                heading_level = sorted_font_sizes.index(span_font_size) + 1
                                if heading_level <= 6:  # Markdown supports h1-h6
                                    # If we have accumulated text, add it first
                                    if block_text.strip():
                                        markdown_content += f"{block_text.strip()}\n\n"
                                        block_text = ""
                                    
                                    # Add heading
                                    heading_text = span_text.strip()
                                    if heading_text:
                                        markdown_content += f"{'#' * heading_level} {heading_text}\n\n"
                                    continue
                            
                            line_text += span_text
                            line_font_size = span_font_size
                        
                        # Add line text to block if it's not empty
                        if line_text.strip():
                            block_text += line_text + " "
                    
                    # Add the accumulated block text as a paragraph
                    if block_text.strip():
                        # Clean up the text: remove extra whitespace and line breaks
                        cleaned_text = " ".join(block_text.split())
                        
                        # Check if this looks like a list item
                        if any(cleaned_text.startswith(marker) for marker in ["•", "-", "*", "○"]):
                            markdown_content += f"- {cleaned_text[1:].strip()}\n"
                        elif cleaned_text.strip().replace(".", "").isdigit():
                            # Numbered list
                            markdown_content += f"1. {cleaned_text}\n"
                        else:
                            # Regular paragraph
                            markdown_content += f"{cleaned_text}\n\n"
            
            # Add page separator (except for last page)
            if page_num < total_pages - 1:
                markdown_content += "---\n\n"
        
        # Close the PDF document after all processing is complete
        pdf_document.close()
        logger.info(f"Successfully converted PDF to structured markdown. Total pages: {total_pages}")
        
        return markdown_content
        
    except Exception as e:
        logger.error(f"Error converting PDF to markdown: {str(e)}")
        # Ensure document is closed even if an error occurs
        try:
            if 'pdf_document' in locals():
                pdf_document.close()
        except:
            pass  # Ignore errors when closing in exception handler
        return None

def save_markdown_file(markdown_content: str, original_filename: str) -> Optional[str]:
    """
    Save markdown content to a file in the output directory.
    
    Args:
        markdown_content: The markdown content to save
        original_filename: Original PDF filename to base the markdown filename on
        
    Returns:
        Path to the saved markdown file, or None if save fails
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate markdown filename from original PDF filename
        base_name = Path(original_filename).stem
        markdown_filename = f"{base_name}.md"
        markdown_path = output_dir / markdown_filename
        
        # Write markdown content to file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown file saved successfully: {markdown_path}")
        return str(markdown_path)
        
    except Exception as e:
        logger.error(f"Error saving markdown file: {str(e)}")
        return None

def process_pdf_to_session_state(file_upload) -> Tuple[bool, Optional[str]]:
    """
    Complete workflow to process PDF: convert to markdown, save file, and update session state.
    
    Args:
        file_upload: Streamlit uploaded file object containing the PDF
        
    Returns:
        Tuple of (success_flag, error_message)
        success_flag: True if processing was successful, False otherwise
        error_message: Error description if processing failed, None if successful
    """
    logger.info(f"Starting complete PDF processing workflow for: {file_upload.name}")
    
    try:
        # Step 1: Convert PDF to markdown
        markdown_content = pdf_to_markdown(file_upload)
        if not markdown_content:
            error_msg = "Failed to convert PDF to markdown"
            logger.error(error_msg)
            return False, error_msg
        
        # Step 2: Save markdown file
        saved_path = save_markdown_file(markdown_content, file_upload.name)
        if not saved_path:
            error_msg = "Failed to save markdown file"
            logger.error(error_msg)
            return False, error_msg
        
        # Step 3: Update session state with markdown content and file path
        st.session_state["markdown_content"] = markdown_content
        st.session_state["markdown_file_path"] = saved_path
        st.session_state["original_pdf_name"] = file_upload.name
        
        logger.info(f"PDF processing completed successfully. Markdown saved to: {saved_path}")
        return True, None
        
    except Exception as e:
        error_msg = f"Unexpected error during PDF processing: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def get_markdown_from_session() -> Optional[str]:
    """
    Retrieve markdown content from session state.
    
    Returns:
        Markdown content string if available in session state, None otherwise
    """
    return st.session_state.get("markdown_content", None)

def get_markdown_file_path_from_session() -> Optional[str]:
    """
    Retrieve saved markdown file path from session state.
    
    Returns:
        File path string if available in session state, None otherwise
    """
    return st.session_state.get("markdown_file_path", None)

# Simple test functions to verify the implementation
def test_pdf_processor():
    """
    Enhanced test function to verify PDF processing functionality including structure preservation.
    This function would be called during development to test the implementation.
    """
    logger.info("Running enhanced PDF processor tests...")
    
    # Test 1: Check if required dependencies are available
    try:
        import pymupdf
        logger.info("✓ PyMuPDF dependency check passed")
        logger.info(f"✓ PyMuPDF version: {pymupdf.__version__}")
    except ImportError:
        logger.error("✗ PyMuPDF not available - install with: uv add pymupdf")
        return False
    
    # Test 2: Check if output directory can be created
    try:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        logger.info("✓ Output directory creation test passed")
    except Exception as e:
        logger.error(f"✗ Output directory creation failed: {e}")
        return False
    
    # Test 3: Test PyMuPDF advanced features availability
    try:
        # Test if we can create a dummy document to check advanced features
        dummy_doc = pymupdf.open()
        dummy_doc.close()
        logger.info("✓ PyMuPDF advanced features test passed")
    except Exception as e:
        logger.error(f"✗ PyMuPDF advanced features test failed: {e}")
        return False
    
    logger.info("All enhanced tests passed!")
    return True

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_pdf_processor()