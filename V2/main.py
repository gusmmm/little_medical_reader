import streamlit as st
import logging
import os
import pdfplumber
from typing import Any, Optional
import sys
from pathlib import Path
import shutil
import re
import tempfile
import zipfile
import datetime
import json

# Configure logging BEFORE any other operations to ensure logger is available
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# streamlit app configuration
st.set_page_config(
    page_title="Little Medical Reader V2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add the parent directory to the Python path to import from file_processor and agents
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules with error handling
try:
    from file_processor.advanced_pdf_processor import AdvancedPdfProcessor, process_pdf_with_docling
    logger.info("✓ Advanced PDF processor imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import advanced PDF processor: {e}")
    st.error("Failed to import advanced PDF processing modules. Please check your installation.")

try:
    from agents.V2_summary_agent import process_medical_article_v2, ArticleAnalysis
    logger.info("✓ V2 Summary Agent imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import V2 Summary Agent: {e}")
    st.error("Failed to import V2 Summary Agent. Please check your installation.")

# use pdfplumber for extracting the pdf pages as images
@st.cache_data
def extract_all_pages_as_images(file_path: str) -> list[Any]:
    """
    Extract all pages of a PDF file as images.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        A list of images extracted from the PDF.
    """
    logger.info(f"Extracting all pages as images from {file_path}.")
    if not os.path.exists(file_path):
        logger.warning("PDF file not found.")
        return []

    images = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                images.append(page.to_image().original)
            logger.info(f"Extracted {len(images)} pages as images from {file_path}.")
    except Exception as e:
        logger.error(f"Error extracting pages from PDF: {e}")
        return []
    
    return images

def extract_and_cache_pdf_pages(file_path: str) -> bool:
    """
    Extract PDF pages and store them in session state to persist across reruns.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        True if extraction was successful, False otherwise
    """
    # Check if we already have pages for this file in session state
    current_file_key = f"pdf_pages_{Path(file_path).name}"
    
    if current_file_key not in st.session_state:
        logger.info(f"Extracting pages for new file: {file_path}")
        images = extract_all_pages_as_images(file_path)
        
        if images:
            st.session_state[current_file_key] = images
            st.session_state["pdf_pages"] = images  # Keep for backward compatibility
            st.session_state["current_pdf_pages_key"] = current_file_key
            logger.info(f"Successfully cached {len(images)} pages for {file_path}")
            return True
        else:
            logger.error(f"Failed to extract pages from {file_path}")
            return False
    else:
        # Pages already cached for this file
        st.session_state["pdf_pages"] = st.session_state[current_file_key]
        st.session_state["current_pdf_pages_key"] = current_file_key
        logger.info(f"Using cached pages for {file_path}")
        return True

def save_uploaded_file_to_input(file_upload) -> Optional[str]:
    """
    Save uploaded file to input directory with collision handling.
    
    Args:
        file_upload: Streamlit uploaded file object
        
    Returns:
        Path to saved file, or None if user chose not to overwrite
    """
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    file_path = input_dir / file_upload.name
    
    # Check if file already exists
    if file_path.exists():
        st.warning(f"⚠️ File '{file_upload.name}' already exists in the input folder.")
        
        col1, col2 = st.columns(2)
        with col1:
            use_existing = st.button(
                "📁 Use Existing File", 
                key="use_existing",
                help="Use the file that's already in the input folder"
            )
        with col2:
            overwrite = st.button(
                "🔄 Overwrite with New File", 
                key="overwrite",
                help="Replace the existing file with the newly uploaded one"
            )
        
        if use_existing:
            logger.info(f"Using existing file: {file_path}")
            return str(file_path)
        elif overwrite:
            logger.info(f"Overwriting existing file: {file_path}")
            with open(file_path, 'wb') as f:
                f.write(file_upload.read())
            return str(file_path)
        else:
            # User hasn't decided yet
            return None
    else:
        # File doesn't exist, save it
        logger.info(f"Saving new file to: {file_path}")
        with open(file_path, 'wb') as f:
            f.write(file_upload.read())
        return str(file_path)

def process_pdf_with_advanced_processor(pdf_path: str) -> Optional[str]:
    """
    Process PDF using the advanced Docling processor.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Path to the output directory for this file, or None if processing fails
    """
    try:
        # Create output directory for this specific file
        pdf_name = Path(pdf_path).stem
        output_dir = Path("output") / pdf_name
        
        logger.info(f"Processing PDF with advanced processor: {pdf_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize advanced processor with specific output directory
        processor = AdvancedPdfProcessor(str(output_dir))
        
        # Process the PDF
        result_path = processor.process_pdf_file(pdf_path)
        
        if result_path:
            logger.info(f"✓ PDF processing successful: {result_path}")
            return str(output_dir)
        else:
            logger.error("✗ PDF processing failed")
            return None
            
    except Exception as e:
        logger.error(f"Error processing PDF with advanced processor: {e}")
        return None

def check_existing_output(pdf_path: str) -> Optional[str]:
    """
    Check if output directory already exists for the given PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Path to existing output directory if it exists, None otherwise
    """
    pdf_name = Path(pdf_path).stem
    output_dir = Path("output") / pdf_name
    
    if output_dir.exists() and output_dir.is_dir():
        # Check if it contains expected files (at least one markdown file)
        md_files = list(output_dir.glob("*.md"))
        if md_files:
            logger.info(f"Found existing output directory: {output_dir}")
            return str(output_dir)
    
    return None

def get_output_summary(output_dir: str) -> dict:
    """
    Get summary statistics of an output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Dictionary with file counts and sizes
    """
    output_path = Path(output_dir)
    
    summary = {
        "md_files": 0,
        "images": 0,
        "tables": 0,
        "total_files": 0,
        "total_size_mb": 0.0,
        "created_time": None
    }
    
    if not output_path.exists():
        return summary
    
    # Count markdown files
    md_files = list(output_path.glob("*.md"))
    summary["md_files"] = len(md_files)
    
    # Count images
    images_dir = output_path / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        summary["images"] = len(image_files)
    
    # Count tables
    tables_dir = output_path / "tables"
    if tables_dir.exists():
        table_files = list(tables_dir.glob("*.png")) + list(tables_dir.glob("*.jpg")) + list(tables_dir.glob("*.jpeg"))
        summary["tables"] = len(table_files)
    
    # Total files and size
    all_files = list(output_path.rglob('*'))
    summary["total_files"] = len([f for f in all_files if f.is_file()])
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    summary["total_size_mb"] = total_size / (1024 * 1024)
    
    # Get creation time of the directory
    try:
        import datetime
        creation_time = output_path.stat().st_mtime
        summary["created_time"] = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        summary["created_time"] = "Unknown"
    
    return summary

def display_processing_results(output_dir: str, pdf_name: str):
    """
    Display the processing results in organized tabs.
    
    Args:
        output_dir: Path to the output directory
        pdf_name: Original PDF filename (without extension)
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        st.error("Output directory not found.")
        return
    
    # Create tabs for different content types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Markdown Files", "🖼️ Extracted Images", "📊 Tables", "🤖 AI Summary", "📁 All Files"])
    
    with tab1:
        st.markdown("### 📄 Generated Markdown Files")
        
        # Look for markdown files
        md_files = list(output_path.glob("*.md"))
        
        if md_files:
            for md_file in md_files:
                with st.expander(f"📝 {md_file.name}", expanded=True):
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Show preview (first 500 characters)
                        st.markdown("**Preview:**")
                        preview = content[:500] + "..." if len(content) > 500 else content
                        st.text(preview)
                        
                        # Show full content in a container
                        with st.container(height=400, border=True):
                            st.markdown(content)
                        
                        # Download button
                        st.download_button(
                            label=f"📥 Download {md_file.name}",
                            data=content,
                            file_name=md_file.name,
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error reading file {md_file.name}: {e}")
        else:
            st.info("No markdown files found in the output directory.")
    
    with tab2:
        st.markdown("### 🖼️ Extracted Images")
        
        # Look for images directory
        images_dir = output_path / "images"
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
            
            if image_files:
                # Helper function to extract page/figure numbers for proper numerical sorting
                def extract_page_number(filename):
                    """Extract page number from filename like 'document_page_12.png' -> 12"""
                    import re
                    match = re.search(r'page_(\d+)', filename)
                    return int(match.group(1)) if match else 0
                
                def extract_figure_number(filename):
                    """Extract figure number from filename like 'document_figure_3.png' -> 3"""
                    import re
                    match = re.search(r'figure_(\d+)', filename)
                    return int(match.group(1)) if match else 0
                
                # Organize images by type and sort numerically by page/figure number
                page_images = sorted([f for f in image_files if "page_" in f.name], key=lambda x: extract_page_number(x.name))
                figure_images = sorted([f for f in image_files if "figure_" in f.name], key=lambda x: extract_figure_number(x.name))
                other_images = sorted([f for f in image_files if "page_" not in f.name and "figure_" not in f.name], key=lambda x: x.name)
                
                # Display page images first
                if page_images:
                    st.markdown("#### 📄 Page Images")
                    st.markdown(f"*{len(page_images)} page images extracted*")
                    
                    # Create pagination for page images (2 per row)
                    images_per_page = 6  # 3 rows of 2 images each
                    total_pages = (len(page_images) + images_per_page - 1) // images_per_page
                    
                    if total_pages > 1:
                        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                        with page_col2:
                            page_num = st.selectbox(
                                "Select page group",
                                range(1, total_pages + 1),
                                format_func=lambda x: f"Pages {(x-1)*images_per_page + 1}-{min(x*images_per_page, len(page_images))}",
                                key="page_images_pagination"
                            )
                    else:
                        page_num = 1
                    
                    # Calculate start and end indices
                    start_idx = (page_num - 1) * images_per_page
                    end_idx = min(start_idx + images_per_page, len(page_images))
                    current_page_images = page_images[start_idx:end_idx]
                    
                    # Display images in pairs (2 per row)
                    for i in range(0, len(current_page_images), 2):
                        col1, col2 = st.columns(2)
                        
                        # First image
                        img_file1 = current_page_images[i]
                        with col1:
                            try:
                                # Create container for image and buttons
                                with st.container(border=True):
                                    st.markdown(f"**{img_file1.name}**")
                                    st.image(str(img_file1), caption=img_file1.name, use_container_width=True)
                                    
                                    # Buttons for image actions
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        # View full size button
                                        if st.button("🔍 Full Size", key=f"fullsize_page_{img_file1.name}", help="View image in full size"):
                                            with st.expander(f"🖼️ Full Size: {img_file1.name}", expanded=True):
                                                st.image(str(img_file1), caption=img_file1.name)
                                    
                                    with btn_col2:
                                        # Download button
                                        with open(img_file1, 'rb') as file:
                                            st.download_button(
                                                label="💾 Save",
                                                data=file.read(),
                                                file_name=img_file1.name,
                                                mime="image/png",
                                                key=f"download_page_{img_file1.name}",
                                                help="Download this image"
                                            )
                            except Exception as e:
                                st.error(f"Error displaying {img_file1.name}: {e}")
                        
                        # Second image (if exists)
                        if i + 1 < len(current_page_images):
                            img_file2 = current_page_images[i + 1]
                            with col2:
                                try:
                                    # Create container for image and buttons
                                    with st.container(border=True):
                                        st.markdown(f"**{img_file2.name}**")
                                        st.image(str(img_file2), caption=img_file2.name, use_container_width=True)
                                        
                                        # Buttons for image actions
                                        btn_col1, btn_col2 = st.columns(2)
                                        with btn_col1:
                                            # View full size button
                                            if st.button("🔍 Full Size", key=f"fullsize_page_{img_file2.name}", help="View image in full size"):
                                                with st.expander(f"🖼️ Full Size: {img_file2.name}", expanded=True):
                                                    st.image(str(img_file2), caption=img_file2.name)
                                        
                                        with btn_col2:
                                            # Download button
                                            with open(img_file2, 'rb') as file:
                                                st.download_button(
                                                    label="💾 Save",
                                                    data=file.read(),
                                                    file_name=img_file2.name,
                                                    mime="image/png",
                                                    key=f"download_page_{img_file2.name}",
                                                    help="Download this image"
                                                )
                                except Exception as e:
                                    st.error(f"Error displaying {img_file2.name}: {e}")
                
                # Display figure images
                if figure_images:
                    st.markdown("---")
                    st.markdown("#### 🖼️ Extracted Figures")
                    st.markdown(f"*{len(figure_images)} figures extracted*")
                    
                    # Display figures in a more compact grid (2 per row, smaller size)
                    for i in range(0, len(figure_images), 2):
                        col1, col2 = st.columns(2)
                        
                        # First figure
                        img_file1 = figure_images[i]
                        with col1:
                            try:
                                with st.container(border=True):
                                    st.markdown(f"**{img_file1.name}**")
                                    # Smaller width for figures to see them more easily
                                    st.image(str(img_file1), caption=img_file1.name, width=300)
                                    
                                    # Buttons for image actions
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        # View full size button
                                        if st.button("🔍 Full Size", key=f"fullsize_fig_{img_file1.name}", help="View figure in full size"):
                                            with st.expander(f"🖼️ Full Size: {img_file1.name}", expanded=True):
                                                st.image(str(img_file1), caption=img_file1.name)
                                    
                                    with btn_col2:
                                        # Download button
                                        with open(img_file1, 'rb') as file:
                                            st.download_button(
                                                label="💾 Save",
                                                data=file.read(),
                                                file_name=img_file1.name,
                                                mime="image/png",
                                                key=f"download_fig_{img_file1.name}",
                                                help="Download this figure"
                                            )
                            except Exception as e:
                                st.error(f"Error displaying {img_file1.name}: {e}")
                        
                        # Second figure (if exists)
                        if i + 1 < len(figure_images):
                            img_file2 = figure_images[i + 1]
                            with col2:
                                try:
                                    with st.container(border=True):
                                        st.markdown(f"**{img_file2.name}**")
                                        # Smaller width for figures to see them more easily
                                        st.image(str(img_file2), caption=img_file2.name, width=300)
                                        
                                        # Buttons for image actions
                                        btn_col1, btn_col2 = st.columns(2)
                                        with btn_col1:
                                            # View full size button
                                            if st.button("🔍 Full Size", key=f"fullsize_fig_{img_file2.name}", help="View figure in full size"):
                                                with st.expander(f"🖼️ Full Size: {img_file2.name}", expanded=True):
                                                    st.image(str(img_file2), caption=img_file2.name)
                                        
                                        with btn_col2:
                                            # Download button
                                            with open(img_file2, 'rb') as file:
                                                st.download_button(
                                                    label="💾 Save",
                                                    data=file.read(),
                                                    file_name=img_file2.name,
                                                    mime="image/png",
                                                    key=f"download_fig_{img_file2.name}",
                                                    help="Download this figure"
                                                )
                                except Exception as e:
                                    st.error(f"Error displaying {img_file2.name}: {e}")
                
                # Display other images if any
                if other_images:
                    st.markdown("---")
                    st.markdown("#### 📎 Other Images")
                    st.markdown(f"*{len(other_images)} other images found*")
                    
                    for i in range(0, len(other_images), 2):
                        col1, col2 = st.columns(2)
                        
                        # First image
                        img_file1 = other_images[i]
                        with col1:
                            try:
                                with st.container(border=True):
                                    st.markdown(f"**{img_file1.name}**")
                                    st.image(str(img_file1), caption=img_file1.name, width=300)
                                    
                                    # Buttons for image actions
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        # View full size button
                                        if st.button("🔍 Full Size", key=f"fullsize_other_{img_file1.name}", help="View image in full size"):
                                            with st.expander(f"🖼️ Full Size: {img_file1.name}", expanded=True):
                                                st.image(str(img_file1), caption=img_file1.name)
                                    
                                    with btn_col2:
                                        # Download button
                                        with open(img_file1, 'rb') as file:
                                            st.download_button(
                                                label="💾 Save",
                                                data=file.read(),
                                                file_name=img_file1.name,
                                                mime="image/png",
                                                key=f"download_other_{img_file1.name}",
                                                help="Download this image"
                                            )
                            except Exception as e:
                                st.error(f"Error displaying {img_file1.name}: {e}")
                        
                        # Second image (if exists)
                        if i + 1 < len(other_images):
                            img_file2 = other_images[i + 1]
                            with col2:
                                try:
                                    with st.container(border=True):
                                        st.markdown(f"**{img_file2.name}**")
                                        st.image(str(img_file2), caption=img_file2.name, width=300)
                                        
                                        # Buttons for image actions
                                        btn_col1, btn_col2 = st.columns(2)
                                        with btn_col1:
                                            # View full size button
                                            if st.button("🔍 Full Size", key=f"fullsize_other_{img_file2.name}", help="View image in full size"):
                                                with st.expander(f"🖼️ Full Size: {img_file2.name}", expanded=True):
                                                    st.image(str(img_file2), caption=img_file2.name)
                                        
                                        with btn_col2:
                                            # Download button
                                            with open(img_file2, 'rb') as file:
                                                st.download_button(
                                                    label="💾 Save",
                                                    data=file.read(),
                                                    file_name=img_file2.name,
                                                    mime="image/png",
                                                    key=f"download_other_{img_file2.name}",
                                                    help="Download this image"
                                                )
                                except Exception as e:
                                    st.error(f"Error displaying {img_file2.name}: {e}")
                
                # Summary at the bottom
                st.markdown("---")
                total_images = len(page_images) + len(figure_images) + len(other_images)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Page Images", len(page_images))
                with col2:
                    st.metric("🖼️ Figure Images", len(figure_images))
                with col3:
                    st.metric("📎 Other Images", len(other_images))
                with col4:
                    st.metric("📊 Total Images", total_images)
                    
                # Bulk download option
                if st.button("📦 Download All Images (ZIP)", help="Download all extracted images as a ZIP file"):
                    try:
                        import zipfile
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for img_file in image_files:
                                    zipf.write(img_file, img_file.name)
                            
                            with open(tmp_file.name, 'rb') as f:
                                zip_data = f.read()
                            
                            st.download_button(
                                label="📥 Download Images ZIP",
                                data=zip_data,
                                file_name=f"{pdf_name}_images.zip",
                                mime="application/zip",
                                key="download_all_images"
                            )
                            
                            os.unlink(tmp_file.name)
                            
                    except Exception as e:
                        st.error(f"Error creating images ZIP file: {e}")
                        
            else:
                st.info("No image files found in the images directory.")
        else:
            st.info("No images directory found.")
    
    with tab3:
        st.markdown("### 📊 Extracted Tables")
        
        # Look for tables directory
        tables_dir = output_path / "tables"
        
        if tables_dir.exists():
            table_files = list(tables_dir.glob("*.png")) + list(tables_dir.glob("*.jpg")) + list(tables_dir.glob("*.jpeg"))
            
            if table_files:
                # Helper function to extract table numbers for proper numerical sorting
                def extract_table_number(filename):
                    """Extract table number from filename like 'document_table_3.png' -> 3"""
                    import re
                    match = re.search(r'table_(\d+)', filename)
                    return int(match.group(1)) if match else 0
                
                # Sort table files numerically by table number
                table_files = sorted(table_files, key=lambda x: extract_table_number(x.name))
                st.markdown(f"*{len(table_files)} tables extracted*")
                
                # Display tables in pairs (2 per row) with interaction
                for i in range(0, len(table_files), 2):
                    col1, col2 = st.columns(2)
                    
                    # First table
                    table_file1 = table_files[i]
                    with col1:
                        try:
                            with st.container(border=True):
                                st.markdown(f"**{table_file1.name}**")
                                # Medium size for tables for better readability
                                st.image(str(table_file1), caption=table_file1.name, width=400)
                                
                                # Buttons for table actions
                                btn_col1, btn_col2 = st.columns(2)
                                with btn_col1:
                                    # View full size button
                                    if st.button("🔍 Full Size", key=f"fullsize_table_{table_file1.name}", help="View table in full size"):
                                        with st.expander(f"📊 Full Size: {table_file1.name}", expanded=True):
                                            st.image(str(table_file1), caption=table_file1.name)
                                
                                with btn_col2:
                                    # Download button
                                    with open(table_file1, 'rb') as file:
                                        st.download_button(
                                            label="💾 Save",
                                            data=file.read(),
                                            file_name=table_file1.name,
                                            mime="image/png",
                                            key=f"download_table_{table_file1.name}",
                                            help="Download this table"
                                        )
                        except Exception as e:
                            st.error(f"Error displaying {table_file1.name}: {e}")
                    
                    # Second table (if exists)
                    if i + 1 < len(table_files):
                        table_file2 = table_files[i + 1]
                        with col2:
                            try:
                                with st.container(border=True):
                                    st.markdown(f"**{table_file2.name}**")
                                    # Medium size for tables for better readability
                                    st.image(str(table_file2), caption=table_file2.name, width=400)
                                    
                                    # Buttons for table actions
                                    btn_col1, btn_col2 = st.columns(2)
                                    with btn_col1:
                                        # View full size button
                                        if st.button("🔍 Full Size", key=f"fullsize_table_{table_file2.name}", help="View table in full size"):
                                            with st.expander(f"📊 Full Size: {table_file2.name}", expanded=True):
                                                st.image(str(table_file2), caption=table_file2.name)
                                    
                                    with btn_col2:
                                        # Download button
                                        with open(table_file2, 'rb') as file:
                                            st.download_button(
                                                label="💾 Save",
                                                data=file.read(),
                                                file_name=table_file2.name,
                                                mime="image/png",
                                                key=f"download_table_{table_file2.name}",
                                                help="Download this table"
                                            )
                            except Exception as e:
                                st.error(f"Error displaying {table_file2.name}: {e}")
                
                # Summary and bulk download
                st.markdown("---")
                st.metric("📊 Total Tables", len(table_files))
                
                # Bulk download option for tables
                if st.button("📦 Download All Tables (ZIP)", help="Download all extracted tables as a ZIP file"):
                    try:
                        import zipfile
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for table_file in table_files:
                                    zipf.write(table_file, table_file.name)
                            
                            with open(tmp_file.name, 'rb') as f:
                                zip_data = f.read()
                            
                            st.download_button(
                                label="📥 Download Tables ZIP",
                                data=zip_data,
                                file_name=f"{pdf_name}_tables.zip",
                                mime="application/zip",
                                key="download_all_tables"
                            )
                            
                            os.unlink(tmp_file.name)
                            
                    except Exception as e:
                        st.error(f"Error creating tables ZIP file: {e}")
                        
            else:
                st.info("No table files found in the tables directory.")
        else:
            st.info("No tables directory found.")
    
    with tab4:
        st.markdown("### 🤖 AI-Powered Article Analysis")
        
        # Check if analysis already exists in session state
        analysis_key = f"ai_analysis_{pdf_name}"
        
        if analysis_key in st.session_state:
            # Display existing analysis
            analysis = st.session_state[analysis_key]
            display_ai_analysis(analysis, output_dir, pdf_name)
        else:
            # Check if analysis file already exists
            analysis_file = Path(output_dir) / f"{pdf_name}_article_analysis.json"
            
            if analysis_file.exists():
                st.info("🔍 Existing AI analysis found!")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📂 Load Existing Analysis", help="Load the previously generated AI analysis"):
                        try:
                            with open(analysis_file, 'r', encoding='utf-8') as f:
                                analysis_data = json.load(f)
                            
                            # Convert to ArticleAnalysis object for consistency
                            from agents.V2_summary_agent import ArticleAnalysis, CitationInfo
                            analysis = ArticleAnalysis(**analysis_data)
                            
                            # Store in session state
                            st.session_state[analysis_key] = analysis
                            st.success("✅ Analysis loaded successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error loading existing analysis: {e}")
                
                with col2:
                    if st.button("🔄 Generate New Analysis", help="Generate a fresh AI analysis"):
                        generate_new_analysis(output_dir, pdf_name, analysis_key)
            else:
                # No existing analysis
                st.info("🤖 Ready to generate AI-powered analysis of this medical article!")
                
                if st.button("🚀 Generate AI Analysis", help="Analyze the article with AI to extract key insights", type="primary"):
                    generate_new_analysis(output_dir, pdf_name, analysis_key)
    
    with tab5:
        st.markdown("### 📁 All Generated Files")
        
        # Show directory structure
        st.markdown("#### 📂 Directory Structure")
        
        def show_directory_tree(path: Path, prefix: str = ""):
            """Recursively show directory structure"""
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                if item.is_dir():
                    st.markdown(f"{prefix}📁 **{item.name}/**")
                    show_directory_tree(item, prefix + "  ")
                else:
                    size = item.stat().st_size
                    size_str = f"({size:,} bytes)" if size < 1024*1024 else f"({size/(1024*1024):.1f} MB)"
                    st.markdown(f"{prefix}📄 {item.name} {size_str}")
        
        show_directory_tree(output_path)
        
        # File operations
        st.markdown("#### 🔧 File Operations")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download All Files (ZIP)", help="Create and download a ZIP file with all generated content"):
                try:
                    import zipfile
                    import tempfile
                    
                    # Create a temporary ZIP file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for file_path in output_path.rglob('*'):
                                if file_path.is_file():
                                    arcname = file_path.relative_to(output_path)
                                    zipf.write(file_path, arcname)
                        
                        # Read the ZIP file for download
                        with open(tmp_file.name, 'rb') as f:
                            zip_data = f.read()
                        
                        st.download_button(
                            label="📦 Download ZIP File",
                            data=zip_data,
                            file_name=f"{pdf_name}_processed_files.zip",
                            mime="application/zip"
                        )
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    st.error(f"Error creating ZIP file: {e}")
        
        with col2:
            total_files = len(list(output_path.rglob('*')))
            total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            st.metric("Total Files", total_files)
            st.metric("Total Size", f"{size_mb:.2f} MB")

def display_ai_analysis(analysis: ArticleAnalysis, output_dir: str, pdf_name: str):
    """
    Display the AI analysis in a beautiful and organized format.
    
    Args:
        analysis: The ArticleAnalysis object from the V2 summary agent.
        output_dir: The output directory for saving the analysis.
        pdf_name: The name of the PDF file.
    """
    st.markdown("#### 🔬 Analysis Results")
    
    # Display Citation Information
    st.markdown(f"**Title:** {analysis.citation_info.title}")
    if analysis.citation_info.first_author:
        authors_text = analysis.citation_info.first_author
        if analysis.citation_info.second_author:
            authors_text += f", {analysis.citation_info.second_author}"
        st.markdown(f"**Authors:** {authors_text}")
    if analysis.citation_info.journal_name:
        st.markdown(f"**Journal:** {analysis.citation_info.journal_name}")
    if analysis.citation_info.publication_date:
        st.markdown(f"**Publication Date:** {analysis.citation_info.publication_date}")
    st.markdown("---")
    
    # Key sections in tabs for better organization
    summary_tab, findings_tab, methods_tab, discussion_tab = st.tabs(
        ["📝 Summary & Keywords", "🔬 Methodology & Results", "� Conclusions", "💬 Discussion"]
    )
    
    with summary_tab:
        st.markdown("##### Abstract Summary")
        st.markdown(analysis.abstract_summary)
        
        st.markdown("##### Keywords")
        if analysis.abstract_keywords:
            keywords_text = ", ".join(analysis.abstract_keywords)
            st.info(f"🔖 {keywords_text}")

    with findings_tab:
        st.markdown("##### 🧪 Methodology")
        st.markdown(analysis.methodology)
        
        st.markdown("##### 📊 Main Results")
        st.markdown(analysis.main_results)

    with methods_tab:
        st.markdown("##### 🎯 Main Conclusions")
        st.markdown(analysis.main_conclusions)
        
        st.markdown("##### ❓ Questions Raised")
        if analysis.questions_raised:
            for question in analysis.questions_raised:
                st.markdown(f"- {question}")
        
        st.markdown("##### � Curiosities")
        if analysis.curiosities:
            for curiosity in analysis.curiosities:
                st.markdown(f"- {curiosity}")

    with discussion_tab:
        st.markdown("##### � Discussion Points")
        if analysis.discussion_points:
            if "good_qualities" in analysis.discussion_points:
                st.markdown("**✅ Strengths:**")
                for quality in analysis.discussion_points["good_qualities"]:
                    st.markdown(f"- {quality}")
            
            if "bad_qualities" in analysis.discussion_points:
                st.markdown("**⚠️ Limitations:**")
                for quality in analysis.discussion_points["bad_qualities"]:
                    st.markdown(f"- {quality}")
        else:
            st.info("No discussion points were extracted.")
            
    # Save analysis button
    st.markdown("---")
    if st.button("💾 Save Analysis to File", help="Save the analysis as a JSON file"):
        try:
            output_path = Path(output_dir)
            analysis_file = output_path / f"{pdf_name}_article_analysis.json"
            
            # Add timestamp to the analysis before saving
            analysis_dict = analysis.model_dump()
            analysis_dict["timestamp_created"] = datetime.datetime.now().isoformat()
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                # Save with timestamp included
                json.dump(analysis_dict, f, indent=4, ensure_ascii=False)
                
            st.success(f"✅ Analysis saved to `{analysis_file}`")
            
            # Provide download button for the saved file
            with open(analysis_file, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="📥 Download Analysis JSON",
                    data=f.read(),
                    file_name=f"{pdf_name}_article_analysis.json",
                    mime="application/json"
                )
        except Exception as e:
            st.error(f"Error saving analysis: {e}")

def generate_new_analysis(output_dir: str, pdf_name: str, analysis_key: str):
    """
    Generate new AI analysis for the article.
    
    Args:
        output_dir: Path to the output directory.
        pdf_name: Name of the PDF file.
        analysis_key: Key for session state.
    """
    st.info("🚀 Generating new AI analysis... This may take a moment.")
    
    # Find the enhanced markdown file to verify it exists
    output_path = Path(output_dir)
    md_file = next(output_path.glob("*_enhanced.md"), None)

    if not md_file:
        st.error("Could not find the enhanced markdown file for analysis.")
        logger.error(f"Enhanced markdown file not found in {output_dir}")
        return

    try:
        logger.info(f"Starting AI analysis for output directory: {output_dir}")
        
        with st.spinner("🤖 AI is analyzing the article... Please wait."):
            # Call the V2 summary agent with the output folder path
            analysis_result = process_medical_article_v2(output_dir)
        
        if analysis_result:
            # Store result in session state
            st.session_state[analysis_key] = analysis_result
            st.success("✅ AI analysis complete!")
            st.rerun()
        else:
            st.error("✗ Failed to generate AI analysis.")
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        logger.error(f"Error during AI analysis generation: {e}")

def main() -> None:
    """
    Main function to run the Streamlit app V2.
    """
    logger.info("Starting Little Medical Reader V2 app")
    
    st.subheader("Welcome to Little Medical Reader V2", divider="grey", anchor=False)
    st.markdown("*Advanced PDF processing with Docling for medical journal articles*")
    
    col1, col2 = st.columns([1.5, 2])

    # File upload and PDF display section
    with col1:
        st.markdown("### 📄 Upload your medical file")
        file_upload = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="file_uploader",
            help="Upload your medical document here."
        )
        
        if file_upload:
            st.success(f"File '{file_upload.name}' uploaded successfully!")
            
            # Save file to input directory with collision handling
            with st.spinner("Saving file to input directory..."):
                saved_file_path = save_uploaded_file_to_input(file_upload)
            
            if saved_file_path:
                st.success(f"✅ File saved to: `{saved_file_path}`")
                
                # Store file path in session state
                st.session_state["current_pdf_path"] = saved_file_path
                st.session_state["current_pdf_name"] = Path(saved_file_path).stem
                
                # Extract and cache PDF pages as images
                with st.spinner("Extracting PDF pages..."):
                    extraction_success = extract_and_cache_pdf_pages(saved_file_path)
                
                if extraction_success:
                    num_pages = len(st.session_state.get("pdf_pages", []))
                    st.success(f"📄 Extracted {num_pages} pages from the file.")
                else:
                    st.error("Failed to extract pages from PDF.")
            else:
                st.info("Please choose whether to use the existing file or overwrite it.")
        
        # Display extracted images if available in session state
        if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
            st.markdown("### 🖼️ PDF Pages Preview")
            
            # Show current file info
            if "current_pdf_path" in st.session_state:
                current_file = Path(st.session_state["current_pdf_path"]).name
                st.info(f"📄 Displaying: {current_file} ({len(st.session_state['pdf_pages'])} pages)")
            
            # Zoom level slider
            zoom_level = st.slider(
                "Zoom Level",
                min_value=100,
                max_value=800,
                value=400,
                step=50,
                key="zoom_slider",
                help="Adjust the zoom level for the PDF page images."
            )
            
            # Display pages in a scrollable container
            with st.container(height=500, border=True):
                for i, page_image in enumerate(st.session_state["pdf_pages"]):
                    try:
                        st.image(
                            page_image,
                            caption=f"Page {i+1}",
                            width=zoom_level,
                        )
                    except Exception as e:
                        st.error(f"Error displaying page {i+1}: {e}")
                        logger.error(f"Error displaying page {i+1}: {e}")
            
            # Add a button to clear cached images if needed
            if st.button("🔄 Reload PDF Pages", help="Clear cached pages and reload them"):
                # Clear the specific cached pages for this file
                current_file_key = st.session_state.get("current_pdf_pages_key")
                if current_file_key and current_file_key in st.session_state:
                    del st.session_state[current_file_key]
                if "pdf_pages" in st.session_state:
                    del st.session_state["pdf_pages"]
                st.rerun()
    
    # Processing and results section
    with col2:
        st.markdown("### 🔬 Advanced PDF Processing")
        
        # Check if we have a PDF ready for processing
        if "current_pdf_path" in st.session_state:
            pdf_path = st.session_state["current_pdf_path"]
            pdf_name = st.session_state["current_pdf_name"]
            
            st.info(f"📁 Ready to process: `{Path(pdf_path).name}`")
            
            # Check if output already exists
            existing_output = check_existing_output(pdf_path)
            
            if existing_output:
                # Show existing output information
                st.warning("⚠️ Previous processing results found!")
                
                # Get summary of existing output
                output_summary = get_output_summary(existing_output)
                
                with st.expander("📊 Existing Processing Results", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📄 Markdown Files", output_summary["md_files"])
                    with col2:
                        st.metric("🖼️ Images", output_summary["images"])
                    with col3:
                        st.metric("📊 Tables", output_summary["tables"])
                    with col4:
                        st.metric("📁 Total Files", output_summary["total_files"])
                    
                    st.markdown(f"**📂 Output Directory:** `{existing_output}`")
                    st.markdown(f"**📅 Created:** {output_summary['created_time']}")
                    st.markdown(f"**💾 Total Size:** {output_summary['total_size_mb']:.2f} MB")
                
                # Provide user options
                st.markdown("### 🤔 What would you like to do?")
                col1, col2 = st.columns(2)
                
                with col1:
                    view_existing = st.button(
                        "👀 View Existing Results",
                        type="primary",
                        key="view_existing",
                        help="Display the previously processed results",
                        use_container_width=True
                    )
                
                with col2:
                    reprocess = st.button(
                        "🔄 Process Again",
                        key="reprocess",
                        help="Delete existing results and process the PDF again",
                        use_container_width=True
                    )
                
                if view_existing:
                    st.success("✅ Loading existing processing results!")
                    st.session_state["processing_output_dir"] = existing_output
                    st.session_state["processing_completed"] = True
                    st.rerun()
                
                elif reprocess:
                    # Confirm deletion and reprocess
                    st.warning("🗑️ This will delete the existing results and reprocess the PDF.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        confirm_delete = st.button(
                            "✅ Confirm Reprocess",
                            key="confirm_reprocess",
                            help="Confirm deletion and start reprocessing"
                        )
                    with col2:
                        cancel_delete = st.button(
                            "❌ Cancel",
                            key="cancel_reprocess",
                            help="Cancel the reprocessing"
                        )
                    
                    if confirm_delete:
                        try:
                            # Delete existing output directory
                            import shutil
                            shutil.rmtree(existing_output)
                            logger.info(f"Deleted existing output directory: {existing_output}")
                            
                            # Process the PDF
                            with st.spinner("🔄 Processing PDF with advanced Docling processor..."):
                                output_dir = process_pdf_with_advanced_processor(pdf_path)
                            
                            if output_dir:
                                st.success("✅ PDF reprocessing completed successfully!")
                                st.session_state["processing_output_dir"] = output_dir
                                st.session_state["processing_completed"] = True
                                st.rerun()
                            else:
                                st.error("❌ PDF reprocessing failed. Please check the logs for details.")
                                
                        except Exception as e:
                            st.error(f"❌ Error during reprocessing: {e}")
                            logger.error(f"Error during reprocessing: {e}")
                    
                    elif cancel_delete:
                        st.info("Reprocessing cancelled.")
                        st.rerun()
            
            else:
                # No existing output, show normal processing button
                if st.button("🚀 Convert to Markdown with Docling", type="primary", help="Process PDF using advanced Docling processor"):
                    with st.spinner("🔄 Processing PDF with advanced Docling processor..."):
                        output_dir = process_pdf_with_advanced_processor(pdf_path)
                    
                    if output_dir:
                        st.success("✅ PDF processing completed successfully!")
                        st.session_state["processing_output_dir"] = output_dir
                        st.session_state["processing_completed"] = True
                        
                        # Show processing summary
                        with st.expander("📊 Processing Summary", expanded=True):
                            output_path = Path(output_dir)
                            
                            # Count generated files
                            md_files = list(output_path.glob("*.md"))
                            image_files = list((output_path / "images").glob("*")) if (output_path / "images").exists() else []
                            table_files = list((output_path / "tables").glob("*")) if (output_path / "tables").exists() else []
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📄 Markdown Files", len(md_files))
                            with col2:
                                st.metric("🖼️ Images Extracted", len(image_files))
                            with col3:
                                st.metric("📊 Tables Extracted", len(table_files))
                            
                            st.markdown(f"**Output Directory:** `{output_dir}`")
                    else:
                        st.error("❌ PDF processing failed. Please check the logs for details.")
            
            # Display processing results if available
            if st.session_state.get("processing_completed", False) and "processing_output_dir" in st.session_state:
                st.markdown("---")
                st.markdown("### 📋 Processing Results")
                
                output_dir = st.session_state["processing_output_dir"]
                display_processing_results(output_dir, pdf_name)
        
        else:
            st.info("📁 Upload a PDF file to begin processing.")
            
            # Show existing processed files in output directory
            output_base_dir = Path("output")
            if output_base_dir.exists():
                existing_folders = [d for d in output_base_dir.iterdir() if d.is_dir() and d.name != "docling_md"]
                
                if existing_folders:
                    st.markdown("### 📚 Previously Processed Files")
                    st.markdown("*You can also view results from previously processed PDFs:*")
                    
                    for folder in existing_folders[:5]:  # Show first 5 folders
                        folder_summary = get_output_summary(str(folder))
                        
                        with st.expander(f"📁 {folder.name}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MD Files", folder_summary["md_files"])
                            with col2:
                                st.metric("Images", folder_summary["images"])
                            with col3:
                                st.metric("Tables", folder_summary["tables"])
                            
                            st.markdown(f"**Created:** {folder_summary['created_time']}")
                            st.markdown(f"**Size:** {folder_summary['total_size_mb']:.2f} MB")
                            
                            if st.button(f"👀 View Results", key=f"view_{folder.name}", help=f"View processing results for {folder.name}"):
                                st.session_state["processing_output_dir"] = str(folder)
                                st.session_state["processing_completed"] = True
                                st.session_state["current_pdf_name"] = folder.name
                                st.rerun()
                    
                    if len(existing_folders) > 5:
                        st.info(f"Showing first 5 of {len(existing_folders)} processed files.")
            
            # Show example of expected output
            st.markdown("### 🎯 What to Expect")
            st.markdown("""
            **Advanced PDF Processing with Docling will generate:**
            
            📄 **Markdown Files:**
            - Enhanced markdown with medical journal structure
            - Properly formatted headings and sections
            - Referenced images and tables
            
            🖼️ **Extracted Images:**
            - High-resolution page images
            - Individual figures and diagrams
            - Medical charts and illustrations
            
            📊 **Table Extraction:**
            - Tables converted to images
            - Preserved formatting and structure
            - Medical data tables and results
            
            🏗️ **Organized Structure:**
            - Dedicated folders for each file type
            - Easy navigation and download options
            - Professional medical document formatting
            """)
            
            # Show example from existing output
            example_dir = Path("output/docling_md")
            if example_dir.exists():
                st.markdown("### 📚 Example Output")
                st.markdown("*Based on previous processing results:*")
                
                # Show example structure
                md_files = list(example_dir.glob("*.md"))
                image_files = list((example_dir / "images").glob("*")) if (example_dir / "images").exists() else []
                table_files = list((example_dir / "tables").glob("*")) if (example_dir / "tables").exists() else []
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Example MD Files", len(md_files))
                with col2:
                    st.metric("Example Images", len(image_files))
                with col3:
                    st.metric("Example Tables", len(table_files))

if __name__ == "__main__":
    main()