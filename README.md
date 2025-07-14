# Little Medical Reader V2

A comprehensive medical document processing and analysis tool that converts medical journal articles and patient notes into structured, analyzable formats with AI-powered insights and natural language processing capabilities.

## Features

The Little Medical Reader V2 is designed to work with medical documents in multiple formats:
- **PDF** - Advanced processing with Docling for superior structure preservation, table extraction, and image handling
- **Markdown** - Direct processing and analysis with enhanced medical structure detection
- **Text** - Direct processing and analysis

The V2 app features a modern six-tab interface built with Streamlit, AI-powered document analysis using Google Gemini 2.0-flash, comprehensive NLP analysis with visualization, and comprehensive medical journal article processing with Docling technology.

## Current Implementation - V2 Features

The V2 version includes major enhancements over the original implementation:

### ✅ V2 Enhanced Features

#### Advanced PDF Processing with Docling
- **Docling Integration**: State-of-the-art PDF processing using Docling technology for superior structure preservation
- **Enhanced Table Extraction**: Automatic detection and conversion of tables to markdown format with visual table images
- **Superior Image Handling**: Extraction of page images, figure images, and other visual elements with organized classification
- **Medical Journal Optimization**: Specialized processing for medical journal articles with section-aware formatting
- **Enhanced Markdown Generation**: Creates both base and enhanced markdown versions with improved medical structure
- **Hierarchical Content**: Automatic detection and formatting of medical sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion)

#### Comprehensive Six-Tab Interface
- **📄 Markdown Files**: View and download generated markdown content with structure preservation
- **🖼️ Extracted Images**: Browse page images, figure images, and other visual elements with full-size viewing and download options
- **📊 Tables**: View extracted tables with enhanced visualization and individual download capabilities
- **🤖 AI Summary**: Generate comprehensive AI-powered article analysis with structured academic formatting
- **📊 NLP Analysis**: Perform advanced natural language processing with visualizations and metrics
- **📁 All Files**: Complete file management with directory structure visualization and bulk download options

#### Advanced NLP Analysis & Visualization
- **Medical NLP Engine**: Comprehensive natural language processing specifically designed for medical documents
- **Text Preprocessing**: Advanced text cleaning with medical-specific stop words and reference removal
- **Meaningful Word Extraction**: Intelligent identification of medically relevant terms with frequency analysis
- **Collocation Analysis**: Detection of bigrams and trigrams for medical term relationships
- **Medical Term Recognition**: Specialized extraction of medical terminology with pattern matching
- **Sentiment Analysis**: Medical document sentiment scoring using VADER lexicon
- **Readability Metrics**: Calculation of text complexity and readability scores for medical content
- **Rich Visualizations**: Generation of word clouds, frequency charts, collocation networks, and medical term analysis
- **Export Capabilities**: Complete NLP results in JSON, CSV, and visualization formats

#### Enhanced AI-Powered Analysis & Summarization
- **V2 Summary Agent**: Advanced medical article analysis with structured data extraction
- **Citation Information Extraction**: Automatic detection of title, authors, journal, DOI, and publication date
- **Abstract Processing**: Intelligent keyword extraction and summary generation from abstracts
- **Methodology Analysis**: Structured extraction of research methods and experimental design
- **Results Interpretation**: Comprehensive analysis of study findings and statistical results
- **Discussion Evaluation**: Assessment of study strengths, limitations, and clinical implications
- **Research Questions**: Generation of relevant follow-up questions based on article content
- **Curiosity Points**: Identification of interesting aspects for further investigation
- **Academic Formatting**: Professional presentation of analysis results in tabbed interface
- **Persistent Analysis**: Save and reload analysis results with JSON export capabilities

#### Medical Document Processing Excellence
- **Smart File Management**: Automatic organization of output files with collision detection and overwrite protection
- **Progress Tracking**: Real-time progress indicators for all processing stages with background continuation
- **Session Persistence**: Maintain processing state across tab navigation and browser sessions
- **Error Recovery**: Robust error handling with detailed logging and user-friendly error messages
- **Bulk Operations**: Download all generated files as organized ZIP archives
- **File Type Recognition**: Intelligent detection and appropriate handling of different file types
- **Output Organization**: Structured output directories with separate folders for images, tables, and analysis results

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

#### Advanced User Interface & Experience
- **Modern Six-Tab Design**: Comprehensive interface with dedicated tabs for different analysis types:
  - **Markdown Files**: View generated markdown with preview and full content display
  - **Extracted Images**: Visual content browser with pagination, full-size viewing, and download options
  - **Tables**: Interactive table viewer with individual access and bulk download capabilities  
  - **AI Summary**: Professional academic analysis presentation with citation information
  - **NLP Analysis**: Comprehensive linguistic analysis with metrics and visualizations
  - **All Files**: Complete file management with directory tree visualization and bulk operations
- **Interactive Analysis Generation**: On-demand processing with comprehensive progress indicators and background processing
- **Enhanced Download Options**: Individual file downloads and bulk ZIP creation for all content types
- **Visual Content Management**: Organized display of page images, figures, and tables with interactive controls
- **Real-time Processing Status**: Live progress tracking with ability to navigate between tabs during processing
- **Session State Management**: Persistent storage of all processing results across browser sessions

#### Enhanced Technical Infrastructure
- **Smart Entry Point**: Intelligent application launcher (`main.py`) with automatic port detection and conflict resolution (8501-8510)
- **Port Conflict Management**: Socket-based port availability checking with graceful fallback and user notification
- **Comprehensive Logging**: Detailed logging system throughout all processing stages for debugging and monitoring
- **Robust Error Handling**: Graceful error handling with user-friendly messages and fallback mechanisms
- **Background Processing**: Continue processing while navigating between tabs with persistent progress tracking
- **API Integration**: Secure integration with Google Gemini AI for document analysis and summarization
- **Modular Architecture**: Clean separation between PDF processing, AI agents, NLP analysis, and UI components
- **Performance Optimization**: Efficient processing with caching and progressive loading for large documents
- **Dependency Management**: Complete dependency resolution with uv package manager integration
- **Application Lifecycle**: Proper startup, shutdown, and error recovery with keyboard interrupt handling

### 🆕 V2 New Capabilities

#### Docling-Powered PDF Processing
- **Superior Structure Preservation**: Advanced document layout analysis with section hierarchy detection
- **Professional Table Extraction**: Convert complex tables to markdown while maintaining visual table images
- **Enhanced Image Classification**: Automatic categorization of page images, figures, and other visual elements
- **Medical Journal Optimization**: Specialized formatting for academic and clinical document standards

#### Natural Language Processing Suite
- **Medical Text Analysis**: Comprehensive NLP pipeline designed specifically for medical content
- **Visualization Dashboard**: Interactive charts and graphs for linguistic analysis results
- **Term Relationship Mapping**: Analysis of medical term collocations and semantic relationships
- **Export-Ready Results**: Complete NLP analysis in multiple formats for further research

#### File Management Excellence
- **Intelligent Organization**: Automatic creation of structured output directories with logical file organization
- **Bulk Operations**: ZIP file creation for easy download and sharing of all generated content
- **Version Control**: Smart handling of file conflicts with user-controlled overwrite options
- **Cross-Session Persistence**: Maintain processing results across application restarts

### 🚧 Future Enhancements
- **Medical Concepts Extraction**: Integration of advanced medical concept extraction for knowledge graphs (from V1)
- **Multi-format Support**: Direct upload and processing of Markdown and Text files
- **Local AI Models**: Integration with Ollama for offline AI processing capabilities
- **Workflow Chains**: LangChain integration for complex document processing workflows
- **Collaboration Features**: Multi-user document review and annotation capabilities
- **Neo4j Integration**: Direct import capabilities for medical knowledge graphs
- **Advanced Analytics**: Enhanced document analytics and insight extraction

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
- `streamlit` - Web framework for the modern six-tab UI interface
- `docling` - Advanced PDF processing with superior structure preservation and table extraction
- `pdfplumber` - PDF image extraction and visual content processing
- `google-genai` - Google Gemini AI integration for document analysis and summarization
- `nltk` - Natural language processing toolkit for medical text analysis
- `matplotlib`, `seaborn`, `wordcloud` - Visualization libraries for NLP analysis results
- `pandas` - Data manipulation and analysis for NLP metrics
- `networkx` - Network analysis for term relationship visualization
- `pydantic` - Data validation for structured AI outputs and clinical data models
- `python-dotenv` - Environment variable management for secure API key handling

3. Set up environment variables:
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

## Running the Application

### Start the V2 Application

To run the Little Medical Reader V2 using the smart entry point:

```bash
uv run main.py
```

This will automatically:
- Launch the V2 Streamlit application with advanced features
- Find an available port (starting from 8501, scanning up to 8510 if needed)
- Display the exact URL where the app is accessible
- Handle port conflicts gracefully with automatic fallback
- Provide comprehensive error handling and logging

The app will start and display something like:
```
🏥 Welcome to Little Medical Reader V2!
📄 Advanced PDF processing for medical journal articles
--------------------------------------------------
🚀 Launching Little Medical Reader V2...
🌐 App will be available at: http://localhost:8501
```

If port 8501 is busy, you'll see:
```
🏥 Welcome to Little Medical Reader V2!
📄 Advanced PDF processing for medical journal articles
--------------------------------------------------
🚀 Launching Little Medical Reader V2...
ℹ️  Port 8501 is busy, using port 8502 instead
🌐 App will be available at: http://localhost:8502
```

### Alternative Methods

You can also run the application directly (though the main entry point is recommended):

```bash
uv run streamlit run V2/main.py
```

Or activate the virtual environment and run manually:

```bash
uv shell
streamlit run V2/main.py
```

**Note**: The main entry point (`uv run main.py`) is strongly recommended as it includes:
- Automatic port conflict detection and resolution
- Comprehensive error handling and user-friendly messages
- Proper logging and debugging information
- Graceful shutdown handling
- Smart fallback mechanisms for robust application startup

## Usage

### V2 Workflow Overview

1. **Launch the Application**: Run `uv run main.py` to start the V2 interface with smart port management
2. **Upload Medical PDF**: Use the file uploader to select a medical journal article or document
3. **Automatic Processing**: The app will process your document using advanced Docling technology
4. **Explore Six Analysis Tabs**: Navigate through different types of analysis and content
5. **Download Results**: Save processed content, analysis results, and visualizations

The V2 application automatically handles:
- **Port Management**: Finds available ports (8501-8510) and handles conflicts gracefully
- **Error Recovery**: Comprehensive error handling with user-friendly messages
- **Background Processing**: Continue working while analysis runs in the background
- **Session Persistence**: Maintains state across browser sessions and tab navigation
- **Smart Logging**: Detailed logs for debugging and monitoring application performance

### Six-Tab Analysis System

The V2 interface provides comprehensive analysis through six dedicated tabs:

#### 📄 Markdown Files Tab
- **View Generated Content**: Browse both base and enhanced markdown versions of your document
- **Structure Preservation**: See how the document structure is maintained with proper medical section formatting
- **Content Preview**: Get a quick preview of the content before viewing the full document
- **Download Options**: Save markdown files for offline use or further processing
- **Medical Formatting**: Enhanced formatting specific to medical journal articles and clinical documents

#### 🖼️ Extracted Images Tab
- **Organized Visual Content**: Browse page images, figure images, and other visual elements in organized categories
- **Pagination Support**: Navigate through large numbers of images with user-friendly pagination
- **Full-Size Viewing**: Click to view any image in full resolution with expandable containers
- **Individual Downloads**: Save specific images with dedicated download buttons
- **Image Classification**: Automatic categorization of different image types for easy navigation
- **Interactive Controls**: Zoom, download, and organize your visual content efficiently

#### 📊 Tables Tab  
- **Extracted Table Display**: View all tables extracted from your document with enhanced visibility
- **Interactive Table Viewer**: See tables in medium resolution with options to view full-size versions
- **Individual Table Access**: Download specific tables as needed for your research
- **Bulk Table Downloads**: Create ZIP files containing all extracted tables
- **Table Classification**: Organized display with proper table numbering and identification

#### 🤖 AI Summary Tab
- **Comprehensive Analysis**: Generate detailed AI-powered analysis using Google Gemini 2.0-flash
- **Academic Structure**: Professional presentation with citation information, methodology, results, and conclusions
- **Tabbed Organization**: Navigate through Summary & Keywords, Methodology & Results, Conclusions, and Discussion sections
- **Citation Extraction**: Automatic detection of title, authors, journal, DOI, and publication information
- **Research Insights**: Generate relevant questions and identify curiosities for further investigation
- **Export Capabilities**: Save complete analysis as structured JSON files for further use

#### 📊 NLP Analysis Tab
- **Advanced Text Processing**: Comprehensive natural language processing specifically designed for medical content using NLTK
- **Medical-Specific Analysis**: Specialized preprocessing with medical stop words, reference removal, and clinical terminology recognition
- **Linguistic Metrics**: Calculate readability scores, lexical diversity, sentiment analysis, and comprehensive text statistics
- **Medical Term Extraction**: Identify and analyze medical terminology with frequency analysis and pattern recognition
- **Visualization Dashboard**: Interactive charts including word clouds, frequency distributions, collocation networks, and medical term analysis
- **Term Relationships**: Analyze bigrams, trigrams, and medical term collocations with network visualization
- **Export Options**: Download complete NLP results as JSON, CSV, and high-quality PNG visualizations
- **Background Processing**: Real-time processing indicators with background continuation capability and session persistence
- **Quality Metrics**: Comprehensive analysis including sentiment analysis, readability scores, and lexical diversity measurements

#### 📁 All Files Tab
- **Complete File Management**: View the entire directory structure of generated content
- **Directory Tree Visualization**: Navigate through organized folder structures with visual representations
- **Bulk Download Operations**: Create and download ZIP files containing all generated content
- **File Organization**: See how content is automatically organized into logical folder structures
- **Storage Management**: Monitor file sizes and manage your generated content efficiently

### Advanced V2 Features

#### Docling-Powered PDF Processing
The V2 version uses state-of-the-art Docling technology for superior PDF processing:

**Enhanced Structure Preservation:**
- Automatic detection of medical journal sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- Hierarchical heading structure with proper markdown formatting
- Preservation of complex document layouts and formatting
- Intelligent paragraph and list detection with medical context awareness

**Advanced Table Processing:**
- Automatic table detection and extraction with visual preservation
- Conversion to structured markdown format while maintaining table images
- Individual table access and download capabilities
- Proper table numbering and organization for research use

**Superior Image Handling:**
- Classification of page images, figure images, and other visual content
- High-quality image extraction with organized folder structure
- Interactive viewing with full-size expansion capabilities
- Bulk download options for efficient content management

#### Natural Language Processing Excellence
The integrated NLP analysis provides comprehensive linguistic insights using the advanced `MedicalNLPAnalyzer`:

**Medical Text Analysis:**
- Specialized preprocessing for medical documents with intelligent reference removal and medical terminology preservation
- Medical-specific stop word filtering with clinical terminology recognition and pattern matching
- Advanced lemmatization and tokenization optimized for clinical content and academic medical writing
- Intelligent meaningful word extraction with frequency analysis and medical relevance scoring
- Named Entity Recognition (NER) for medical entities and relationships

**Visualization and Metrics:**
- Professional word clouds and frequency distribution charts with medical-optimized styling
- Collocation network analysis for term relationships and semantic connections
- Medical term frequency analysis with specialized pattern recognition and clinical relevance scoring  
- Readability metrics (Flesch-like scoring) and sentiment analysis specifically tuned for medical content
- Comprehensive text quality metrics including lexical diversity, sentence complexity, and vocabulary richness

**Export and Integration:**
- Complete results in structured JSON format for further analysis and research integration
- CSV export for spreadsheet compatibility and statistical analysis
- High-quality PNG visualizations (300 DPI) for presentations, reports, and publications
- Organized ZIP archives for easy sharing, archival, and collaborative research workflows
- Background processing with session persistence for large document analysis

### Session Persistence and Performance

**Background Processing:**
- Continue analysis while navigating between tabs
- Persistent session state across browser refreshes
- Intelligent caching of processing results for quick access
- Real-time progress indicators with detailed status updates

**File Management:**
- Automatic organization of output files in logical directory structures
- Smart collision detection with user-controlled overwrite options
- Complete file history and version tracking
- Efficient storage management with cleanup options

**Performance Optimization:**
- Progressive loading for large documents
- Efficient memory management for processing large PDFs
- Optimized image handling and display
- Background processing that doesn't block user interface interactions

## Project Structure

```
little_medical_reader/
├── main.py                  # Smart entry point with port management and error handling
├── V2/
│   ├── __init__.py
│   └── main.py              # V2 Streamlit app with six-tab interface and advanced features
├── file_processor/
│   ├── __init__.py
│   ├── pdf_processor.py     # Original PDF processing (V1)
│   └── advanced_pdf_processor.py  # Docling-powered PDF processing for V2
├── agents/
│   ├── __init__.py
│   ├── summary_agent.py     # V1 AI-powered document analysis and summarization
│   ├── concepts_agent.py    # Medical concepts extraction for knowledge graphs (V1)
│   └── V2_summary_agent.py  # Enhanced V2 AI analysis with structured academic formatting
├── NLP/
│   ├── __init__.py
│   ├── basic_nlp.py        # Basic NLP functionality (V1)
│   ├── advanced_nlp.py     # Advanced NLP processing (V1)
│   └── V2/
│       ├── __init__.py
│       └── nlp.py          # Comprehensive medical NLP analyzer for V2
├── UI/                     # Original V1 interface (legacy)
│   ├── __init__.py
│   └── main.py            # V1 Streamlit app with three-tab interface
├── output/                 # Generated files (created automatically)
│   ├── <document_name>/   # Individual document folders with organized content
│   │   ├── *.md          # Base and enhanced markdown files
│   │   ├── *_article_analysis.json  # AI analysis results
│   │   ├── images/       # Page images, figures, and other visual content
│   │   ├── tables/       # Extracted table images
│   │   └── nlp/          # NLP analysis results with visualizations
│   └── docling_md/       # Legacy output directory
├── input/                 # Input files directory (created automatically)
├── logs/                  # Application logs (created automatically)
├── .env                   # Environment variables (create manually)
├── main.py               # V2 application entry point with smart port management
├── pyproject.toml        # Project dependencies and metadata with V2 requirements
├── uv.lock              # Dependency lock file
├── README.md            # This file - V2 documentation
└── LICENSE              # Project license
```

## Development

The V2 app architecture provides enhanced separation of concerns:

- **V2 UI Layer**: Modern six-tab Streamlit interface with advanced document processing and analysis capabilities
- **Advanced Processing Layer**: Docling-powered PDF processing with superior structure preservation and content extraction
- **Enhanced AI Layer**: V2 summary agent with structured academic analysis and comprehensive result formatting  
- **NLP Analysis Layer**: Comprehensive medical natural language processing with visualization and export capabilities
- **File Management Layer**: Intelligent organization with automatic directory structures and bulk operations
- **Legacy Support**: V1 components maintained for backward compatibility and feature migration

### V2 Architecture Improvements

- **Modular Design**: Clean separation between processing, analysis, and presentation layers
- **Background Processing**: Non-blocking operations that continue while users navigate between tabs
- **Session Management**: Persistent state management across browser sessions and application restarts
- **Error Recovery**: Comprehensive error handling with graceful degradation and user feedback
- **Performance Optimization**: Efficient processing with caching and progressive loading for large documents
- **Export Systems**: Multiple export formats with organized file structures for research and sharing

## Technology Stack

- **Backend**: Python 3.12+ with comprehensive medical document processing capabilities
- **UI Framework**: Streamlit with modern six-tab interface for comprehensive document analysis
- **Advanced PDF Processing**: Docling integration for superior structure preservation, table extraction, and content organization
- **Legacy PDF Processing**: PyMuPDF and pdfplumber for compatibility and image extraction
- **AI Integration**: Google Gemini 2.0-flash for document analysis, academic summarization, and structured content extraction
- **NLP Processing**: NLTK ecosystem with medical-specific enhancements for comprehensive text analysis
- **Data Visualization**: Matplotlib, Seaborn, WordCloud, and NetworkX for rich NLP analysis visualizations
- **Data Management**: Pandas for data manipulation and analysis with CSV export capabilities
- **Data Validation**: Pydantic v2 for structured AI outputs and clinical data models
- **Package Management**: uv for fast dependency management and development workflow
- **Environment Management**: python-dotenv for secure API key handling and configuration
- **Export Capabilities**: JSON, CSV, PNG, and ZIP formats for research and sharing
- **Future Integrations**: LangChain (planned), Ollama (planned), Neo4j medical knowledge graphs (planned)

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

The application provides comprehensive logging across all V2 modules:
- **PDF Processing**: Track Docling processing progress and advanced content extraction
- **AI Analysis**: Monitor V2 summary agent processing with structured academic analysis
- **NLP Processing**: Detailed logging of medical text analysis with metrics and visualization generation
- **File Management**: Track automatic organization and bulk operations
- **Background Processing**: Monitor progress of operations that continue while navigating between tabs
- **Error Handling**: Detailed error logging with context for troubleshooting V2 features
- **Performance**: Track processing times and resource usage for large document processing

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

**Port Conflicts:**
The new smart entry point automatically handles port conflicts. If you see "Port already in use" errors:
- The application will automatically try ports 8501-8510
- You'll see a notification showing which port is being used
- The exact URL will be displayed in the console

**Application Won't Start:**
```bash
# Check if uv is installed
uv --version

# Install dependencies
uv sync

# Try running with verbose output
uv run main.py
```

**AI Features Not Working**: 
- Ensure `GEMINI_API_KEY` is set in `.env` file
- Check internet connectivity for API access
- Verify API key has sufficient quota

**Medical Concepts Extraction Slow**:
- Large documents are processed in chunks with progress tracking
- Processing continues in background even when switching tabs
- Check logs for detailed progress information

**PDF Processing Errors**:
- Check that uploaded file is a valid PDF
- Ensure the PDF is not password-protected
- Check that the file size is reasonable (< 100MB recommended)
- Verify the PDF contains extractable text (not just scanned images)
- Ensure sufficient disk space for processing
- Review logs in `logs/` directory for detailed error information

**NLP Analysis Errors:**
- The application automatically downloads required NLTK data
- Ensure sufficient disk space for visualization generation
- Check logs in the `logs/` directory for detailed error information

**Performance Issues**:
- Large PDFs may take longer to process
- AI processing requires internet connectivity
- Medical concepts extraction is limited to 50 most relevant concepts for performance

### Log Analysis

Check the application logs for detailed troubleshooting information:
- Location: `logs/` directory (created automatically)
- Format: Timestamped entries with module identification and detailed context
- Levels: INFO, WARNING, ERROR for different severity levels with comprehensive stack traces
- Content: PDF processing, AI analysis, NLP processing, concept extraction, port management, and user interactions
- Performance: Processing times, memory usage, and optimization recommendations

### Entry Point Debugging

The smart entry point provides comprehensive error handling:
- **Port Discovery**: Automatic scanning and conflict resolution with detailed logging
- **Path Validation**: Verification of V2 app location and dependency availability
- **Error Recovery**: Graceful fallback mechanisms with user-friendly error messages
- **Process Management**: Proper subprocess handling with cleanup on interruption
- **Logging Integration**: Detailed startup and shutdown logging for debugging application lifecycle

## License

See LICENSE file for details.

---

**Note**: This application is designed for healthcare professionals and should be used as a supplementary tool. Always verify AI-generated summaries and extracted medical concepts, and maintain clinical judgment in patient care decisions. The medical concepts extraction is intended to support knowledge organization and should not replace professional medical analysis.