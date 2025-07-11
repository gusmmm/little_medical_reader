# Little Medical Reader

This app helps you read your patient's notes by converting the content to a visual format after extracting the data.

## Features

The Little Medical Reader is designed to work with patient notes in three formats:
- **PDF** - The app will convert it to markdown
- **Markdown** - Direct processing
- **Text** - Direct processing

The app features a graphical interface built with Streamlit and will use workflow chains with LLM and AI agents powered by LangChain.

## Current Implementation

The current version of the app includes:

### ✅ Implemented Features
- **PDF Upload Interface**: Upload PDF medical documents through a user-friendly Streamlit interface
- **PDF Page Extraction**: Extract all pages from PDF files as images using pdfplumber for visual inspection
- **PDF to Markdown Conversion**: Convert PDF content to markdown format using PyMuPDF for text extraction and processing
- **Dual Content Display**: View both visual pages (images) and converted text content (markdown) side by side
- **Markdown File Download**: Download the converted markdown content as a file for offline use
- **Session State Management**: Maintain extracted pages and converted content in session state for persistent viewing
- **File Processing Pipeline**: Complete workflow from PDF upload to markdown conversion with proper error handling
- **Responsive Layout**: Two-column layout with file upload and images on the left, markdown content on the right
- **Progress Indicators**: Loading spinners and success/error messages for user feedback during processing
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring PDF processing operations

### 🚧 Planned Features
- Text and Markdown file processing (direct upload)
- LLM integration with Google Gemini 2.0 Flash for content analysis
- Local model support with Ollama for offline processing
- AI-powered content analysis and visualization
- Workflow chains using LangChain for advanced document processing

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites
- Python 3.12 or higher
- uv package manager

### Install Dependencies

1. Clone the repository:
```bash
git clone <repository-url>
cd little_medical_reader
```

2. Install dependencies using uv:
```bash
uv sync
```

This will install all required dependencies including:
- `streamlit` - Web framework for the UI
- `pdfplumber` - PDF processing and image extraction
- `pymupdf` - PDF text extraction and markdown conversion

## Running the Application

### Start the Streamlit App

To run the Streamlit application using uv:

```bash
uv run streamlit run UI/main.py
```

The app will start on `http://localhost:8501` by default.

### Alternative Method

You can also activate the virtual environment and run streamlit directly:

```bash
uv shell
streamlit run UI/main.py
```

## Usage

1. **Upload a PDF**: Use the file uploader in the left column to select a PDF medical document
2. **Automatic Processing**: The app will automatically:
   - Extract all pages as images for visual inspection
   - Convert the PDF content to markdown format for text analysis
3. **View Content**: 
   - **Left Column**: Browse through extracted page images with adjustable zoom
   - **Right Column**: Read the converted markdown content in a scrollable container
4. **Download**: Use the download button to save the converted markdown file to your local machine
5. **Session Persistence**: Your processed content remains available during your session

## Project Structure

```
little_medical_reader/
├── UI/
│   ├── __init__.py
│   └── main.py          # Streamlit app main file
├── file_processor/
│   ├── __init__.py
│   └── pdf_processor.py # PDF processing and markdown conversion
├── output/              # Generated markdown files (created automatically)
├── main.py              # Project entry point
├── pyproject.toml       # Project dependencies and metadata
├── uv.lock             # Dependency lock file
├── README.md           # This file
└── LICENSE             # Project license
```

## Development

The app is structured to support:
- **UI Layer**: Streamlit-based interface in the `UI/` folder
- **Processing Layer**: PDF and document processing in the `file_processor/` folder
- **Workflow Chains**: Future LangChain integration for AI processing
- **Multiple Model Support**: Google Gemini 2.0 Flash for cloud processing, Ollama for local processing

## Technology Stack

- **Backend**: Python 3.12+
- **UI Framework**: Streamlit
- **PDF Processing**: pdfplumber (images), PyMuPDF (text extraction)
- **Package Management**: uv
- **Future AI Integration**: LangChain, Google Gemini, Ollama

## Contributing

When contributing to this project:
1. Use `uv` for dependency management
2. Follow the coding instructions in `.github/copilot-instructions.md`
3. Implement features in small, verifiable steps
4. Add comprehensive code annotations
5. Include tests for new features

## License

See LICENSE file for details.