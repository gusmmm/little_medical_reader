"""
Advanced PDF Processor using Docling for medical journal articles.

This module provides enhanced PDF processing capabilities with:
- Image extraction and embedding
- Table detection and conversion to markdown
- Better text structure preservation
- Medical journal specific formatting
- Section-based organization using headings and subheadings

Author: GitHub Copilot
Date: July 13, 2025
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import streamlit as st
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

# Configure logging for this module
logger = logging.getLogger(__name__)

class AdvancedPdfProcessor:
    """
    Advanced PDF processor for medical journal articles using Docling.
    
    This class provides comprehensive PDF processing with:
    - Automatic table detection and conversion
    - Image extraction and markdown embedding
    - Hierarchical heading structure preservation
    - Medical journal specific formatting
    """
    
    def __init__(self, output_dir: str = "output/docling_md"):
        """
        Initialize the advanced PDF processor.
        
        Args:
            output_dir: Directory to save processed markdown and extracted assets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for assets
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        self.images_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        # Configure Docling converter with proper options for figure extraction
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.images_scale = 2.0  # Higher resolution for extracted images
        self.pipeline_options.generate_page_images = True  # Enable page image generation
        self.pipeline_options.generate_picture_images = True  # Enable picture extraction
        
        # Initialize document converter with format options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        logger.info(f"Advanced PDF processor initialized with output directory: {self.output_dir}")
    
    def process_pdf_file(self, pdf_path: str) -> Optional[str]:
        """
        Process a PDF file and convert it to structured markdown with images and tables.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Path to the generated markdown file, or None if processing fails
        """
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            logger.info(f"Starting advanced PDF processing for: {pdf_file.name}")
            
            # Convert PDF using Docling
            conv_res = self.converter.convert(pdf_path)
            
            if not conv_res.document:
                logger.error("Failed to convert PDF document")
                return None
            
            doc_filename = conv_res.input.file.stem
            
            # Extract and save images and tables
            self._extract_images_and_tables(conv_res, doc_filename)
            
            # Generate structured markdown using Docling's built-in method
            markdown_path = self.output_dir / f"{doc_filename}_docling.md"
            conv_res.document.save_as_markdown(
                markdown_path, 
                image_mode=ImageRefMode.REFERENCED
            )
            
            # Enhance the markdown with medical journal structure
            enhanced_content = self._enhance_medical_structure(markdown_path, doc_filename)
            
            # Save enhanced markdown
            enhanced_path = self.output_dir / f"{doc_filename}_enhanced.md"
            with open(enhanced_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            
            logger.info(f"Successfully processed PDF to markdown: {enhanced_path}")
            return str(enhanced_path)
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            return None
    
    def process_pdf_upload(self, file_upload) -> Optional[str]:
        """
        Process an uploaded PDF file from Streamlit.
        
        Args:
            file_upload: Streamlit uploaded file object
            
        Returns:
            Path to the generated markdown file, or None if processing fails
        """
        try:
            logger.info(f"Processing uploaded PDF: {file_upload.name}")
            
            # Save uploaded file temporarily
            temp_path = self.output_dir / f"temp_{file_upload.name}"
            with open(temp_path, 'wb') as f:
                f.write(file_upload.read())
            
            # Process the temporary file
            result_path = self.process_pdf_file(str(temp_path))
            
            # Clean up temporary file
            temp_path.unlink()
            
            return result_path
            
        except Exception as e:
            logger.error(f"Error processing uploaded PDF: {str(e)}")
            return None
    
    def _extract_images_and_tables(self, conv_res, doc_filename: str):
        """
        Extract and save images and tables from the document.
        
        Args:
            conv_res: Docling conversion result
            doc_filename: Base filename for saving assets
        """
        logger.info("Extracting images and tables...")
        
        # Save page images if available
        for page_no, page in conv_res.document.pages.items():
            if hasattr(page, 'image') and page.image:
                page_image_filename = self.images_dir / f"{doc_filename}_page_{page_no}.png"
                try:
                    with page_image_filename.open("wb") as fp:
                        page.image.pil_image.save(fp, format="PNG")
                    logger.info(f"Saved page image: {page_image_filename}")
                except Exception as e:
                    logger.warning(f"Could not save page {page_no} image: {e}")
        
        # Extract tables and figures
        table_counter = 0
        picture_counter = 0
        
        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                table_image_filename = self.tables_dir / f"{doc_filename}_table_{table_counter}.png"
                try:
                    with table_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")
                    logger.info(f"Saved table image: {table_image_filename}")
                except Exception as e:
                    logger.warning(f"Could not save table {table_counter}: {e}")
            
            if isinstance(element, PictureItem):
                picture_counter += 1
                picture_image_filename = self.images_dir / f"{doc_filename}_figure_{picture_counter}.png"
                try:
                    with picture_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")
                    logger.info(f"Saved figure image: {picture_image_filename}")
                except Exception as e:
                    logger.warning(f"Could not save figure {picture_counter}: {e}")
        
        logger.info(f"Extracted {table_counter} tables and {picture_counter} figures")
    
    def _enhance_medical_structure(self, markdown_path: Path, doc_filename: str) -> str:
        """
        Enhance the markdown with medical journal specific structure and formatting.
        
        Args:
            markdown_path: Path to the base markdown file
            doc_filename: Document filename for metadata
            
        Returns:
            Enhanced markdown content
        """
        try:
            # Read the base markdown
            with open(markdown_path, 'r', encoding='utf-8') as f:
                base_content = f.read()
            
            # Create enhanced header
            enhanced_lines = [
                f"# {doc_filename.replace('_', ' ').replace('-', ' ').title()}",
                "",
                "---",
                "",
                "## Document Information",
                "",
                f"- **Source**: PDF Document processed with Docling",
                f"- **Processing Date**: {self._get_current_date()}",
                f"- **Images Directory**: `images/`",
                f"- **Tables Directory**: `tables/`",
                "",
                "---",
                ""
            ]
            
            # Process the base content to improve medical journal formatting
            content_lines = base_content.split('\n')
            processed_lines = []
            
            for line in content_lines:
                # Enhance medical section headings
                if line.strip() and any(keyword in line.lower() for keyword in 
                    ['abstract', 'introduction', 'methods', 'methodology', 'results', 
                     'discussion', 'conclusion', 'references', 'background']):
                    if not line.startswith('#'):
                        processed_lines.append(f"## {line.strip()}")
                    else:
                        processed_lines.append(line)
                
                # Improve subsection formatting
                elif line.strip() and any(line.strip().startswith(prefix) for prefix in 
                    ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']):
                    processed_lines.append(f"### {line.strip()}")
                
                # Keep other lines as is
                else:
                    processed_lines.append(line)
            
            # Combine header with processed content
            return '\n'.join(enhanced_lines + processed_lines)
            
        except Exception as e:
            logger.error(f"Error enhancing medical structure: {e}")
            return f"# Error Processing Document\n\nCould not enhance markdown structure: {e}"
    
    def _get_current_date(self) -> str:
        """
        Get current date in a readable format.
        
        Returns:
            Formatted date string
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Convenience functions for integration with existing codebase
def process_pdf_with_docling(pdf_path: str, output_dir: str = "output/docling_md") -> Optional[str]:
    """
    Process a PDF file using Docling advanced processor.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for processed files
        
    Returns:
        Path to generated markdown file, or None if processing fails
    """
    processor = AdvancedPdfProcessor(output_dir)
    return processor.process_pdf_file(pdf_path)

def process_pdf_upload_with_docling(file_upload, output_dir: str = "output/docling_md") -> Optional[str]:
    """
    Process an uploaded PDF file using Docling advanced processor.
    
    Args:
        file_upload: Streamlit uploaded file object
        output_dir: Output directory for processed files
        
    Returns:
        Path to generated markdown file, or None if processing fails
    """
    processor = AdvancedPdfProcessor(output_dir)
    return processor.process_pdf_upload(file_upload)

def update_session_state_with_docling(markdown_path: str, original_filename: str):
    """
    Update Streamlit session state with Docling processing results.
    
    Args:
        markdown_path: Path to generated markdown file
        original_filename: Original PDF filename
    """
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        st.session_state["docling_markdown_content"] = markdown_content
        st.session_state["docling_markdown_path"] = markdown_path
        st.session_state["original_pdf_name"] = original_filename
        
        logger.info(f"Session state updated with Docling results: {markdown_path}")
        
    except Exception as e:
        logger.error(f"Error updating session state: {str(e)}")

# Test function
def test_advanced_pdf_processor():
    """
    Test the advanced PDF processor functionality.
    """
    logger.info("Testing Advanced PDF Processor with Docling...")
    
    try:
        # Test processor initialization
        processor = AdvancedPdfProcessor()
        logger.info("✓ Processor initialization successful")
        
        # Check if example PDF exists
        example_pdf = Path("input/jcm-12-03188.pdf")
        if example_pdf.exists():
            logger.info(f"✓ Found example PDF: {example_pdf}")
            
            # Test processing
            result = processor.process_pdf_file(str(example_pdf))
            if result:
                logger.info(f"✓ PDF processing successful: {result}")
            else:
                logger.error("✗ PDF processing failed")
                return False
        else:
            logger.warning("⚠ Example PDF not found - skipping processing test")
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run tests
    test_advanced_pdf_processor()
