# Import necessary libraries
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import json
import logging
import os
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
def download_nltk_data():
    """
    Downloads the necessary NLTK data files.
    """
    resources = {
        'tokenizers/punkt': 'punkt',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
        'chunkers/maxent_ne_chunker': 'maxent_ne_chunker',
        'corpora/words': 'words',
        'corpora/wordnet': 'wordnet',
        'corpora/stopwords': 'stopwords',
        'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng',
        'chunkers/maxent_ne_chunker_tab': 'maxent_ne_chunker_tab'
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logging.info(f"Downloading '{resource_name}'...")
            nltk.download(resource_name)

# Perform advanced NLP analysis
def advanced_nlp_analysis(text):
    """
    Performs advanced NLP analysis on the given text to extract semantic information,
    focusing on medical text by removing stopwords and irrelevant terms.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        dict: A dictionary containing the extracted semantic data.
    """
    logging.info("Starting advanced NLP analysis for medical text...")
    
    # Ensure NLTK data is available
    download_nltk_data()

    # --- Text Pre-processing ---
    # Remove URLs and references like [1,2]
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[[0-9, ]+\]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text.lower()) # Convert to lower case for better stopword matching
    
    # Define custom stopwords and extend the default list
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {
        'et', 'al', 'fig', 'figure', 'table', 'author', 'copyright', 'licensee', 'mdpi', 
        'basel', 'switzerland', 'article', 'journal', 'https', 'doi', 'org', 'www', 
        'page', 'j', 'clin', 'med', 'cont', 'note', 'tid', 'bid', 'qid', 'ld'
    }
    stop_words.update(custom_stopwords)
    
    # Filter out stopwords and non-alphabetic tokens
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    logging.info(f"Filtered text from {len(words)} to {len(filtered_words)} words.")

    # Lemmatization with progress tracking and optimization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    # Process in chunks to avoid memory issues and provide progress updates
    chunk_size = 1000
    total_chunks = len(filtered_words) // chunk_size + 1
    
    logging.info(f"Starting lemmatization of {len(filtered_words)} words in {total_chunks} chunks...")
    
    for i in range(0, len(filtered_words), chunk_size):
        chunk = filtered_words[i:i + chunk_size]
        # Use a set to track already lemmatized words to speed up the process
        lemmatized_chunk = [lemmatizer.lemmatize(word) for word in chunk]
        lemmatized_words.extend(lemmatized_chunk)
        
        # Progress logging
        current_chunk = (i // chunk_size) + 1
        if current_chunk % 5 == 0 or current_chunk == total_chunks:
            logging.info(f"Processed chunk {current_chunk}/{total_chunks}")
    
    logging.info("Lemmatization completed.")

    # --- Analysis on Processed Text ---
    # Re-join for sentence-level analysis if needed, though we focus on terms now
    processed_text = ' '.join(lemmatized_words)

    # Part-of-Speech (POS) tagging on the lemmatized words
    pos_tagged_words = pos_tag(lemmatized_words)
    
    # Named Entity Recognition (NER) - on original case sentences for better accuracy
    # We will use the original text for NER as casing is important
    # Limit to first 50 sentences to avoid performance issues
    sentences_orig = sent_tokenize(text)[:50]  # Limit for performance
    logging.info(f"Processing NER on first {len(sentences_orig)} sentences...")
    
    words_orig = [word_tokenize(sent) for sent in sentences_orig]
    pos_tagged_orig = [pos_tag(word) for word in words_orig]

    named_entities = []
    for i, tagged_sentence in enumerate(pos_tagged_orig):
        tree = ne_chunk(tagged_sentence)
        for subtree in tree.subtrees(filter=lambda t: t.label() != 'S'):
            entity = " ".join([word for word, tag in subtree.leaves()])
            named_entities.append({'entity': entity, 'label': subtree.label()})
        
        # Progress logging for NER
        if (i + 1) % 10 == 0:
            logging.info(f"Processed NER for {i + 1}/{len(pos_tagged_orig)} sentences")
            
    logging.info(f"Found {len(named_entities)} named entities.")
    
    # Extract key terms (nouns, adjectives, and verbs)
    key_terms = [word for word, tag in pos_tagged_words if tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('VB')]
                    
    logging.info(f"Extracted {len(key_terms)} key terms.")
    
    # Prepare data for output
    analysis_results = {
        'named_entities': named_entities,
        'key_terms': key_terms, # These are now lemmatized and filtered
        'processed_text': processed_text
    }
    
    logging.info("Advanced NLP analysis completed.")
    
    return analysis_results

# Create visual representation of the analysis
def create_visual_representation(analysis_results, output_folder, filename_prefix):
    """
    Creates and saves more relevant visual representations for medical text analysis.
    
    Args:
        analysis_results (dict): The results from the advanced_nlp_analysis function.
        output_folder (str): The folder to save the visualizations in.
        filename_prefix (str): The prefix for the output filenames.
    """
    logging.info("Creating enhanced visual representations...")
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 1. Word Cloud of Key Terms
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(analysis_results['processed_text'])
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Most Frequent Terms in the Medical Journal')
    wordcloud_path = os.path.join(output_folder, f"{filename_prefix}_wordcloud.png")
    plt.savefig(wordcloud_path)
    plt.close()
    logging.info(f"Saved Word Cloud to {wordcloud_path}")

    # 2. Bar chart of Top 20 Most Frequent Key Terms
    key_term_counts = Counter(analysis_results['key_terms'])
    most_common_terms = key_term_counts.most_common(20)
    
    if most_common_terms:
        plt.figure(figsize=(12, 8))
        plt.barh([item[0] for item in most_common_terms], [item[1] for item in most_common_terms])
        plt.gca().invert_yaxis()
        plt.title('Top 20 Most Frequent Key Terms')
        plt.xlabel('Frequency')
        plt.tight_layout()
        key_terms_chart_path = os.path.join(output_folder, f"{filename_prefix}_key_terms_frequency.png")
        plt.savefig(key_terms_chart_path)
        plt.close()
        logging.info(f"Saved key terms frequency chart to {key_terms_chart_path}")

    # 3. Bar chart of Named Entity types (filtered for relevance)
    entity_labels = [ne['label'] for ne in analysis_results['named_entities']]
    entity_counts = Counter(entity_labels)
    
    if entity_counts:
        plt.figure(figsize=(10, 6))
        plt.bar(entity_counts.keys(), entity_counts.values())
        plt.title('Frequency of Named Entity Types')
        plt.xlabel('Entity Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        ner_chart_path = os.path.join(output_folder, f"{filename_prefix}_ner_distribution.png")
        plt.savefig(ner_chart_path)
        plt.close()
        logging.info(f"Saved NER distribution chart to {ner_chart_path}")

    logging.info("Visual representations created.")

# --- Test Block ---
if __name__ == '__main__':
    logging.info("Running advanced_nlp.py as a standalone script for testing.")
    
    # Define file paths relative to the project root
    input_file_path = os.path.join('output', 'jcm-12-03188.md')
    output_folder = 'output'
    filename_prefix = "medical_journal_nlp"
    output_json_path = os.path.join(output_folder, f"{filename_prefix}_analysis.json")

    # Check if input file exists
    if not os.path.exists(input_file_path):
        logging.error(f"Input file not found at: {input_file_path}")
        logging.error("Please ensure the file 'jcm-12-03188.md' exists in the 'output' directory.")
    else:
        # Read the content of the markdown file
        logging.info(f"Reading content from {input_file_path}")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
            
        # Perform the analysis
        results = advanced_nlp_analysis(markdown_content)
        
        # Save the results to a JSON file (excluding the full processed text for brevity)
        output_data = {k: v for k, v in results.items() if k != 'processed_text'}
        logging.info(f"Saving analysis results to {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
            
        # Create and save visualizations
        create_visual_representation(results, output_folder, filename_prefix)
        
        logging.info("Script finished successfully.")
        # Simple test verification
        print("\n--- Test Verification ---")
        print(f"Analysis complete. Check the following files in the '{output_folder}' directory:")
        print(f"1. Semantic analysis data: {os.path.basename(output_json_path)}")
        print(f"2. Word Cloud: {filename_prefix}_wordcloud.png")
        print(f"3. Key Terms Frequency Chart: {filename_prefix}_key_terms_frequency.png")
        print(f"4. NER Distribution Chart: {filename_prefix}_ner_distribution.png")
        print("------------------------")
