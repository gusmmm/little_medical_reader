import streamlit as st
import logging
import os
import pdfplumber
from typing import Any, Optional
import sys
from pathlib import Path
import shutil

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
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Markdown Files", "🖼️ Extracted Images", "📊 Tables", "📁 All Files"])
    
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
                # Organize images by type
                page_images = [f for f in image_files if "page_" in f.name]
                figure_images = [f for f in image_files if "figure_" in f.name]
                
                # Display page images
                if page_images:
                    st.markdown("#### 📄 Page Images")
                    cols = st.columns(3)
                    for i, img_file in enumerate(page_images[:9]):  # Show first 9 page images
                        with cols[i % 3]:
                            try:
                                st.image(str(img_file), caption=img_file.name, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying {img_file.name}: {e}")
                    
                    if len(page_images) > 9:
                        st.info(f"Showing first 9 of {len(page_images)} page images.")
                
                # Display figure images
                if figure_images:
                    st.markdown("#### 🖼️ Extracted Figures")
                    for img_file in figure_images:
                        st.markdown(f"**{img_file.name}**")
                        try:
                            st.image(str(img_file), caption=img_file.name, width=600)
                        except Exception as e:
                            st.error(f"Error displaying {img_file.name}: {e}")
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
                for table_file in table_files:
                    st.markdown(f"**{table_file.name}**")
                    try:
                        st.image(str(table_file), caption=table_file.name, width=800)
                    except Exception as e:
                        st.error(f"Error displaying {table_file.name}: {e}")
            else:
                st.info("No table files found in the tables directory.")
        else:
            st.info("No tables directory found.")
    
    with tab4:
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
            
            # Processing button
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