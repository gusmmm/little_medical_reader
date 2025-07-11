# Little Medical Reader

This app helps you read your patient's notes by converting the content to a visual format after extracting the data, and provides AI-powered intelligent summarization and medical concepts extraction for healthcare professionals.

## Features

The Little Medical Reader is designed to work with patient notes in three formats:
- **PDF** - The app will convert it to markdown with structure preservation
- **Markdown** - Direct processing and analysis
- **Text** - Direct processing and analysis

The app features a graphical interface built with Streamlit, AI-powered document analysis using Google Gemini 2.0-flash, comprehensive medical concepts extraction for knowledge graphs, and detailed logging throughout the processing pipeline.

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

#### Medical Concepts Extraction & Knowledge Graphs
- **Advanced Medical Concepts Extraction**: AI-powered identification of medical concepts according to four specific categories:
  - **Individual Concepts**: Specific entities (diseases, medications, patients, anatomical structures)
  - **Qualitative Concepts**: Medical classifications and categories (conditions, findings, assessments)
  - **Comparative Concepts**: Medical comparisons (temporal, severity, reference comparisons)
  - **Quantitative Concepts**: Numerical measurements (vital signs, lab values, measurements)
- **Clinical Relationship Mapping**: Extraction of medically relevant relationships between concepts:
  - Clinical relationships: increases/decreases, improves/worsens, causes, treats
  - Diagnostic relationships: indicates, contraindicates, measures, diagnoses
  - Structural relationships: is_part_of, influences, associated_with
- **Intelligent Concept Filtering**: Advanced relevance scoring that selects the 50 most important concepts based on:
  - Text frequency (30%) - How often concepts appear in the source document
  - Clinical significance (25%) - Medical importance level (critical, high, moderate, low)
  - AI confidence (20%) - Extraction confidence scores
  - Concept type priority (15%) - Clinical value of different concept types
  - Category priority (10%) - Medical category importance
- **Neo4j-Ready Output**: Structured JSON export optimized for medical knowledge graph creation
- **Progress Tracking**: Real-time progress indicators with persistent processing that continues even when switching tabs
- **Clinical Metadata**: Rich metadata including medical specialties, clinical significance levels, and relationship strengths

#### User Interface & Experience
- **Tabbed Interface**: Three dedicated tabs for different analysis types:
  - **Original Content**: View converted markdown with download options
  - **AI Summary**: Generate and view intelligent medical summaries
  - **Medical Concepts**: Extract and explore medical concepts with relationships
- **Interactive Analysis Generation**: On-demand AI processing with comprehensive progress indicators
- **Dual Content Display**: View both visual pages (images) and converted text content (markdown) side by side
- **Document Analysis Display**: Show detected document type, medical specialty, and analysis metadata
- **Comprehensive Download Options**: Download markdown files, AI summaries, and medical concepts JSON
- **Responsive Layout**: Optimized two-column layout for medical document review
- **Real-time Progress Tracking**: Visual progress bars and status updates during AI processing

#### Technical Infrastructure
- **Comprehensive Logging**: Detailed logging system throughout all processing stages for debugging and monitoring
- **Robust Error Handling**: Graceful error handling with user-friendly messages and fallback mechanisms
- **Progress Persistence**: Background processing that continues even when users navigate between tabs
- **API Integration**: Secure integration with Google Gemini AI for document analysis, summarization, and concept extraction
- **Modular Architecture**: Clean separation between PDF processing, AI agents, and UI components

### 🚧 Planned Features
- **Multi-format Support**: Direct upload and processing of Markdown and Text files
- **Local AI Models**: Integration with Ollama for offline AI processing capabilities
- **Advanced Analytics**: Enhanced document analytics and insight extraction
- **Workflow Chains**: LangChain integration for complex document processing workflows
- **Collaboration Features**: Multi-user document review and annotation capabilities
- **Neo4j Integration**: Direct import capabilities for medical knowledge graphs

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
- `google-genai` - Google Gemini AI integration for analysis and concept extraction
- `pydantic` - Data validation for structured AI outputs
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
   - **Right Column**: Access three analysis tabs for different types of content analysis

### Three-Tab Analysis System

#### 📄 Original Content Tab
- View the converted markdown content in a scrollable container
- Download the structured markdown file for offline use
- Review the document structure and formatting preservation

#### 🤖 AI Summary Tab
- **Generate AI Summary**: Click to create an intelligent analysis of your document
- **Document Analysis**: View automatic detection of:
  - Document type (journal article, clinical note, discharge summary, etc.)
  - Medical specialty (cardiology, neurology, oncology, etc.)
  - Complexity level and target audience
- **Adaptive Summaries**: Receive structured summaries tailored to document type:
  - Research articles: Study overview, methodology, key findings, clinical implications
  - Clinical documents: Patient information, key findings, treatment plans, follow-up requirements
  - Case reports: Case presentation, clinical course, outcomes, learning points
- **Download Options**: Save AI-generated summaries as markdown files

#### 🧠 Medical Concepts Tab
- **Extract Medical Concepts**: Click to analyze your document for medical concepts and relationships
- **Progress Tracking**: Real-time progress indicators with the ability to safely navigate between tabs
- **Comprehensive Analysis**: View extracted concepts organized by:
  - **Medical Concept Types**: Individual, qualitative, comparative, quantitative
  - **Medical Categories**: Diseases, medications, symptoms, procedures, lab values, etc.
  - **Clinical Relationships**: How concepts relate to each other with strength scores
  - **Medical Specialties**: Relevant healthcare domains identified
- **Intelligent Filtering**: Automatically selects the 50 most relevant concepts based on clinical importance
- **Neo4j Export**: Download structured JSON data ready for medical knowledge graph creation

### Advanced AI Features

#### Medical Concepts Extraction
The AI system identifies and categorizes medical concepts according to a clinically-relevant taxonomy:

**Concept Types:**
- **Individual**: Specific entities (e.g., "acute myocardial infarction", "metformin 500mg")
- **Qualitative**: Classifications (e.g., "hypertension", "positive blood culture")
- **Comparative**: Comparisons (e.g., "worse than yesterday", "above normal range")
- **Quantitative**: Measurements (e.g., "blood pressure 140/90 mmHg", "hemoglobin 8.2 g/dL")

**Clinical Relationships:**
- Treatment relationships: treats, improves, worsens
- Causal relationships: causes, influences
- Diagnostic relationships: indicates, measures, diagnoses
- Structural relationships: is_part_of, associated_with

#### Intelligent Relevance Filtering
The system automatically prioritizes concepts using a sophisticated scoring algorithm that considers:
- **Frequency in text** (30%): How often the concept appears
- **Clinical significance** (25%): Medical importance level
- **AI confidence** (20%): Extraction confidence score
- **Concept type priority** (15%): Clinical value of concept type
- **Category priority** (10%): Medical category importance

### Session Persistence

Your processed content remains available during your session, allowing you to:
- Switch between original content, AI summaries, and medical concepts
- Navigate away from processing tabs while AI work continues in background
- Regenerate summaries or concepts with different focus
- Download multiple versions of processed documents

## Project Structure

```
little_medical_reader/
├── UI/
│   ├── __init__.py
│   └── main.py              # Streamlit app with integrated AI analysis and concepts extraction
├── file_processor/
│   ├── __init__.py
│   └── pdf_processor.py     # Advanced PDF processing with structure preservation
├── agents/
│   ├── __init__.py
│   ├── summary_agent.py     # AI-powered document analysis and summarization
│   └── concepts_agent.py    # Medical concepts extraction for knowledge graphs
├── output/                  # Generated files (created automatically)
│   ├── *.md                 # Converted markdown files
│   ├── *_AI_summary.md      # AI-generated summaries
│   └── *_medical_concepts.json # Medical concepts for Neo4j import
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
- **UI Layer**: Streamlit-based interface with integrated AI features and tabbed navigation
- **Processing Layer**: Advanced PDF and document processing in the `file_processor/` folder
- **AI Layer**: Document analysis, summarization, and medical concepts extraction in the `agents/` folder
- **Output Management**: Automatic file organization with structured JSON for knowledge graphs
- **Logging**: Comprehensive logging system for debugging and monitoring

## Technology Stack

- **Backend**: Python 3.12+
- **UI Framework**: Streamlit with three-tab interface for different analysis types
- **PDF Processing**: pdfplumber (images), PyMuPDF (advanced text extraction)
- **AI Integration**: Google Gemini 2.0-flash for document analysis, summarization, and medical concepts extraction
- **Data Validation**: Pydantic v2 for structured AI outputs and clinical data models
- **Package Management**: uv for fast dependency management
- **Environment Management**: python-dotenv for secure API key handling
- **Knowledge Graphs**: JSON export optimized for Neo4j medical knowledge graph creation
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
- **AI Analysis**: Monitor document classification, summary generation, and concept extraction
- **Concept Filtering**: Log relevance scoring and filtering decisions
- **Progress Tracking**: Monitor AI processing progress and user interactions
- **Error Handling**: Detailed error logging with context for troubleshooting
- **Performance**: Track processing times and resource usage

Logs are automatically created in the `logs/` directory with timestamps and structured formatting.

## Medical Concepts & Knowledge Graphs

### Neo4j Integration

The medical concepts extraction generates JSON files optimized for Neo4j import:

```json
{
  "extraction_metadata": {
    "source_file": "document.pdf",
    "total_concepts": 50,
    "total_relationships": 45,
    "extraction_model": "gemini-2.0-flash",
    "filtering_applied": "top_50_most_relevant"
  },
  "medical_concepts": [
    {
      "concept_id": "acute-myocardial-infarction",
      "concept_name": "acute myocardial infarction",
      "concept_type": "individual",
      "concept_category": "disease",
      "relationship_type": "causes",
      "relationship_strength": 0.9,
      "clinical_significance": "critical",
      "medical_specialty": "cardiology"
    }
  ],
  "schema_info": {
    "concept_types": ["individual", "qualitative", "comparative", "quantitative"],
    "relationship_types": ["causes", "treats", "indicates", "measures", ...],
    "filtering_info": {
      "max_concepts": 50,
      "filtering_criteria": ["frequency_in_text (30%)", "clinical_significance (25%)", ...]
    }
  }
}
```

### Knowledge Graph Applications

The extracted medical concepts can be used for:
- **Clinical Decision Support**: Link related symptoms, diseases, and treatments
- **Medical Research**: Identify patterns and relationships in medical literature
- **Educational Tools**: Create interactive medical concept maps
- **Quality Improvement**: Analyze clinical documentation for completeness
- **Differential Diagnosis**: Support diagnostic reasoning through concept relationships

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
- **Medical Accuracy**: Ensure clinical relevance and accuracy in medical concept extraction

## Troubleshooting

### Common Issues

1. **AI Features Not Working**: 
   - Ensure `GEMINI_API_KEY` is set in `.env` file
   - Check internet connectivity for API access
   - Verify API key has sufficient quota

2. **Medical Concepts Extraction Slow**:
   - Large documents are processed in chunks with progress tracking
   - Processing continues in background even when switching tabs
   - Check logs for detailed progress information

3. **PDF Processing Errors**:
   - Check that uploaded file is a valid PDF
   - Ensure sufficient disk space for processing
   - Review logs in `logs/` directory for detailed error information

4. **Performance Issues**:
   - Large PDFs may take longer to process
   - AI processing requires internet connectivity
   - Medical concepts extraction is limited to 50 most relevant concepts for performance

### Log Analysis

Check the application logs for detailed troubleshooting information:
- Location: `logs/` directory
- Format: Timestamped entries with module identification
- Levels: INFO, WARNING, ERROR for different severity levels
- Content: PDF processing, AI analysis, concept extraction, and user interactions

## License

See LICENSE file for details.

---

**Note**: This application is designed for healthcare professionals and should be used as a supplementary tool. Always verify AI-generated summaries and extracted medical concepts, and maintain clinical judgment in patient care decisions. The medical concepts extraction is intended to support knowledge organization and should not replace professional medical analysis.