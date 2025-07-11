# Little Medical Reader

This app helps you read your patient's notes by converting the content to a visual format after extracting the data, and provides AI-powered intelligent summarization for healthcare professionals.

## Features

The Little Medical Reader is designed to work with patient notes in three formats:
- **PDF** - The app will convert it to markdown with structure preservation
- **Markdown** - Direct processing and analysis
- **Text** - Direct processing and analysis

The app features a graphical interface built with Streamlit, AI-powered document analysis using Google Gemini 2.0-flash, and comprehensive logging throughout the processing pipeline.

## Current Implementation

The current version of the app includes:

### ✅ Implemented Features

#### Core Document Processing
- **PDF Upload Interface**: Upload PDF medical documents through a user-friendly Streamlit interface
- **Advanced PDF Processing**: Extract PDF content to markdown using PyMuPDF with structure preservation, including:
  - Table of contents extraction and navigation
  - Font size analysis for heading hierarchy detection
  - Text block organization and paragraph detection
  - List item identification and formatting
- **PDF Page Extraction**: Extract all pages from PDF files as images using pdfplumber for visual inspection
- **Session State Management**: Maintain extracted pages and converted content in session state for persistent viewing

#### AI-Powered Analysis & Summarization
- **Document Type Detection**: Intelligent classification of medical documents (journal articles, discharge summaries, clinical notes, lab results, etc.)
- **Adaptive AI Summarization**: Generate structured summaries tailored to document type and target audience using Google Gemini 2.0-flash
- **Medical Specialty Recognition**: Automatically identify relevant medical specialties and adjust summary focus accordingly
- **Multiple Summary Templates**: Different summary structures for research articles, clinical documents, and case reports
- **Fallback Analysis**: Keyword-based document classification when AI analysis is unavailable

#### User Interface & Experience
- **Dual Content Display**: View both visual pages (images) and converted text content (markdown) side by side
- **Tabbed Interface**: Separate tabs for original content and AI-generated summaries
- **Interactive Summary Generation**: On-demand AI summary creation with progress indicators
- **Document Analysis Display**: Show detected document type, medical specialty, and analysis metadata
- **File Download Options**: Download both converted markdown and AI-generated summaries
- **Responsive Layout**: Two-column layout optimized for medical document review

#### Technical Infrastructure
- **Comprehensive Logging**: Detailed logging system throughout all processing stages for debugging and monitoring
- **Error Handling**: Robust error handling with user-friendly error messages and fallback mechanisms
- **Progress Indicators**: Loading spinners and success/error messages for user feedback during processing
- **API Integration**: Secure integration with Google Gemini AI for document analysis and summarization

### 🚧 Planned Features
- **Multi-format Support**: Direct upload and processing of Markdown and Text files
- **Local AI Models**: Integration with Ollama for offline AI processing capabilities
- **Advanced Analytics**: Enhanced document analytics and insight extraction
- **Workflow Chains**: LangChain integration for complex document processing workflows
- **Collaboration Features**: Multi-user document review and annotation capabilities

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites
- Python 3.12 or higher
- uv package manager
- Google Gemini API key (for AI features)

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
- `pymupdf` - Advanced PDF text extraction and markdown conversion
- `google-genai` - Google Gemini AI integration
- `python-dotenv` - Environment variable management

3. Set up environment variables:
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

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

### Basic Workflow

1. **Upload a PDF**: Use the file uploader in the left column to select a PDF medical document
2. **Automatic Processing**: The app will automatically:
   - Extract all pages as images for visual inspection
   - Convert the PDF content to structured markdown format
   - Analyze document structure and preserve hierarchical elements
3. **View Content**: 
   - **Left Column**: Browse through extracted page images with adjustable zoom
   - **Right Column**: Access original markdown content and AI-generated summaries via tabs
4. **Generate AI Summary**: 
   - Click "Generate AI Summary" to create an intelligent analysis
   - View document type detection and medical specialty identification
   - Access structured summary appropriate for healthcare professionals
5. **Download Results**: Save both converted markdown files and AI summaries to your local machine

### AI Features

The AI summarization system provides:
- **Document Classification**: Automatic detection of document types (research papers, clinical notes, discharge summaries, etc.)
- **Specialty Recognition**: Identification of relevant medical specialties (cardiology, neurology, oncology, etc.)
- **Adaptive Summaries**: Different summary structures based on document type:
  - Research articles: Study overview, methodology, key findings, clinical implications
  - Clinical documents: Patient information, key findings, treatment plans, follow-up requirements
  - Case reports: Case presentation, clinical course, outcomes, learning points
- **Target Audience Optimization**: Summaries tailored for specific healthcare professional audiences

### Session Persistence

Your processed content remains available during your session, allowing you to:
- Switch between original content and AI summaries
- Regenerate summaries with different focus
- Download multiple versions of processed documents

## Project Structure

```
little_medical_reader/
├── UI/
│   ├── __init__.py
│   └── main.py              # Streamlit app with AI integration
├── file_processor/
│   ├── __init__.py
│   └── pdf_processor.py     # Advanced PDF processing with structure preservation
├── agents/
│   ├── __init__.py
│   └── summary_agent.py     # AI-powered document analysis and summarization
├── output/                  # Generated markdown and summary files (created automatically)
├── logs/                    # Application logs (created automatically)
├── .env                     # Environment variables (create manually)
├── main.py                  # Project entry point
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # Dependency lock file
├── README.md                # This file
└── LICENSE                  # Project license
```

## Development

The app is structured with clear separation of concerns:
- **UI Layer**: Streamlit-based interface in the `UI/` folder with integrated AI features
- **Processing Layer**: Advanced PDF and document processing in the `file_processor/` folder
- **AI Layer**: Document analysis and summarization agents in the `agents/` folder
- **Output Management**: Automatic file organization in the `output/` folder
- **Logging**: Comprehensive logging system in the `logs/` folder

## Technology Stack

- **Backend**: Python 3.12+
- **UI Framework**: Streamlit with tabbed interface
- **PDF Processing**: pdfplumber (images), PyMuPDF (advanced text extraction)
- **AI Integration**: Google Gemini 2.0-flash for document analysis and summarization
- **Package Management**: uv for fast dependency management
- **Environment Management**: python-dotenv for secure API key handling
- **Future Integrations**: LangChain (planned), Ollama (planned)

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required for AI features
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Logging

The application provides comprehensive logging across all modules:
- **PDF Processing**: Track extraction progress and identify processing issues
- **AI Analysis**: Monitor document classification and summary generation
- **Error Handling**: Detailed error logging with context for troubleshooting
- **Performance**: Track processing times and resource usage

Logs are automatically created in the `logs/` directory with timestamps and structured formatting.

## Contributing

When contributing to this project:
1. Use `uv` for dependency management
2. Follow the coding instructions in `.github/copilot-instructions.md`
3. Implement features in small, verifiable steps
4. Add comprehensive code annotations and logging
5. Include tests for new features
6. Ensure proper error handling and user feedback
7. Update documentation for new features

### Development Guidelines
- **Logging**: Add detailed logging to all new functions
- **Error Handling**: Implement robust error handling with user-friendly messages
- **Testing**: Create tests for new functionality
- **Documentation**: Update README and inline documentation for new features

## Troubleshooting

### Common Issues

1. **AI Features Not Working**: 
   - Ensure `GEMINI_API_KEY` is set in `.env` file
   - Check internet connectivity for API access
   - Verify API key has sufficient quota

2. **PDF Processing Errors**:
   - Check that uploaded file is a valid PDF
   - Ensure sufficient disk space for processing
   - Review logs in `logs/` directory for detailed error information

3. **Performance Issues**:
   - Large PDFs may take longer to process
   - AI summarization requires internet connectivity
   - Check system resources during processing

### Log Analysis

Check the application logs for detailed troubleshooting information:
- Location: `logs/` directory
- Format: Timestamped entries with module identification
- Levels: INFO, WARNING, ERROR for different severity levels

## License

See LICENSE file for details.

---

**Note**: This application is designed for healthcare professionals and should be used as a supplementary tool. Always verify AI-generated summaries and maintain clinical judgment in patient care decisions.