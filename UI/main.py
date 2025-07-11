import streamlit as st
import logging
import os
import pdfplumber
from typing import Any


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
                

if __name__ == "__main__":
    main()