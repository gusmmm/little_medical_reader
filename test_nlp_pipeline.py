#!/usr/bin/env python3
"""
Test script for the biomedical NLP pipeline.
This script tests the NLP pipeline with the example document from jcm-12-03188.

Author: Little Medical Reader Team
Date: 2025-01-27
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from NLP.V2.nlp import MedicalNLPEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nlp_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_nlp_pipeline():
    """
    Test the biomedical NLP pipeline with the example document.
    
    This function demonstrates the capabilities of the NLP pipeline by:
    1. Loading a medical document
    2. Processing it through the pipeline
    3. Extracting insights and generating reports
    4. Creating visualizations
    """
    logger.info("Starting NLP pipeline test...")
    
    try:
        # Initialize the NLP engine
        logger.info("Initializing Medical NLP Engine...")
        nlp_engine = MedicalNLPEngine()
        
        # Path to the test document
        test_document_path = "output/jcm-12-03188/jcm-12-03188_enhanced.md"
        
        if not os.path.exists(test_document_path):
            logger.error(f"Test document not found: {test_document_path}")
            return False
            
        logger.info(f"Loading test document: {test_document_path}")
        
        # Read the document content
        with open(test_document_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
            
        logger.info(f"Document loaded successfully. Length: {len(document_content)} characters")
        
        # Process the document
        logger.info("Processing document through NLP pipeline...")
        analysis_results = nlp_engine.process_documents([document_content])
        
        logger.info("Document processing completed successfully!")
        
        # Extract quick insights
        logger.info("Extracting quick insights...")
        insights = nlp_engine.extract_quick_insights(document_path)
        
        # Print basic statistics
        print("\n" + "="*80)
        print("BIOMEDICAL NLP PIPELINE TEST RESULTS")
        print("="*80)
        
        print(f"\nDocument Statistics:")
        print(f"- Character count: {len(document_content):,}")
        print(f"- Word count: {len(document_content.split()):,}")
        print(f"- Line count: {len(document_content.splitlines()):,}")
        
        # Display analysis results summary
        if analysis_results:
            result = analysis_results
            
            print(f"\nEntity Extraction Results:")
            entities = result.get('entities', {})
            for entity_type, entity_list in entities.items():
                print(f"- {entity_type}: {len(entity_list)} entities")
                # Show first few entities as examples
                if entity_list:
                    examples = entity_list[:3]
                    print(f"  Examples: {', '.join(examples)}")
            
            print(f"\nText Processing Results:")
            print(f"- Sections identified: {len(result.get('sections', []))}")
            print(f"- Medical abbreviations found: {len(result.get('abbreviations', []))}")
            print(f"- Drug dosages detected: {len(result.get('dosages', []))}")
            
        # Display insights
        if insights:
            print(f"\nQuick Insights:")
            for insight_type, insight_data in insights.items():
                if isinstance(insight_data, list):
                    print(f"- {insight_type}: {len(insight_data)} items")
                elif isinstance(insight_data, dict):
                    print(f"- {insight_type}: {len(insight_data)} categories")
                else:
                    print(f"- {insight_type}: {insight_data}")
        
        print(f"\nOutput Files Generated:")
        output_dir = Path("output/nlp_test")
        if output_dir.exists():
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    print(f"- {file_path.name}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        logger.info("NLP pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during NLP pipeline test: {str(e)}")
        logger.exception("Full exception details:")
        print(f"\nERROR: {str(e)}")
        return False

def main():
    """
    Main function to run the NLP pipeline test.
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    print("Biomedical NLP Pipeline Test")
    print("=" * 40)
    
    success = test_nlp_pipeline()
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
