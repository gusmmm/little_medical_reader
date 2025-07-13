"""
Simple NLP Analysis for Medical Documents using NLTK
Analyzes meaningful words, collocations, and relations from medical texts.
Generates metrics and visualization graphs.

Author: GitHub Copilot
Date: 2025-07-13
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import networkx as nx
from collections import Counter, defaultdict
import json

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class MedicalNLPAnalyzer:
    """
    A comprehensive NLP analyzer for medical documents using NLTK.
    """
    
    def __init__(self, output_dir: str = "nlp_output"):
        """
        Initialize the NLP analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Add medical-specific stop words
        self.medical_stop_words = {
            'patient', 'patients', 'study', 'studies', 'treatment', 'treatments',
            'clinical', 'hospital', 'medical', 'therapy', 'disease', 'condition',
            'table', 'figure', 'doi', 'et', 'al', 'mg', 'ml', 'kg', 'h', 'min',
            'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years'
        }
        self.stop_words.update(self.medical_stop_words)
        
        logger.info(f"Initialized MedicalNLPAnalyzer with output directory: {output_dir}")

    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)

    def clean_markdown_text(self, text: str) -> str:
        """
        Clean markdown text by removing references and formatting.
        
        Args:
            text: Raw markdown text
            
        Returns:
            Cleaned text without references and markdown formatting
        """
        logger.info("Cleaning markdown text...")
        original_length = len(text)
        
        # Remove everything after "References" or "Disclaimer" heading
        reference_patterns = [
            r'\n##?\s*(References|Bibliography|Citations|Disclaimer).*',
            r'\n##?\s*(References|Bibliography|Citations|Disclaimer)[\s\S]*$'
        ]
        
        for pattern in reference_patterns:
            before_removal = len(text)
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
            if len(text) < before_removal:
                logger.info(f"Removed references section using pattern: {pattern}")
                logger.info(f"Text reduced from {before_removal} to {len(text)} characters")
                break
        
        # Additional cleanup for numbered references that might remain
        # Remove numbered citation patterns like "1. Author, Title... [CrossRef]"
        text = re.sub(r'\n\d+\.\s+[^\n]*?\[CrossRef\][^\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n\d+\.\s+[^\n]*?\[PubMed\][^\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n\d+\.\s+[^\n]*?\[Google Scholar\][^\n]*', '', text, flags=re.MULTILINE)
        
        # Remove markdown formatting
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Images
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)   # Links
        text = re.sub(r'#{1,6}\s*', '', text)        # Headers
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # Bold/italic
        text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)    # Code
        text = re.sub(r'\|.*?\|', '', text)          # Tables
        text = re.sub(r'---+', '', text)             # Horizontal rules
        text = re.sub(r'^\s*[-\*\+]\s+', '', text, flags=re.MULTILINE)  # Lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)    # Numbered lists
        
        # Clean extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        final_length = len(text)
        logger.info(f"Text cleaning completed. Original: {original_length} chars, Final: {final_length} chars")
        logger.info(f"Removed {original_length - final_length} characters ({((original_length - final_length) / original_length * 100):.1f}%)")
        return text.strip()

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        logger.info("Preprocessing text...")
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter tokens: remove punctuation, numbers, and stop words
        filtered_tokens = []
        for token in tokens:
            # Keep only alphabetic tokens longer than 2 characters
            if (token.isalpha() and 
                len(token) > 2 and 
                token not in self.stop_words):
                # Lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)
        
        logger.info(f"Preprocessed {len(tokens)} tokens to {len(filtered_tokens)} meaningful tokens")
        return filtered_tokens

    def extract_meaningful_words(self, tokens: List[str], top_n: int = 50) -> Dict[str, int]:
        """
        Extract the most meaningful words with their frequencies.
        
        Args:
            tokens: Preprocessed tokens
            top_n: Number of top words to return
            
        Returns:
            Dictionary of word frequencies
        """
        logger.info("Extracting meaningful words...")
        
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Filter out very common and very rare words
        total_words = len(tokens)
        meaningful_words = {}
        
        for word, freq in word_freq.items():
            # Keep words that appear at least 3 times but not more than 20% of total
            if 3 <= freq <= total_words * 0.2:
                meaningful_words[word] = freq
        
        # Sort by frequency and return top N
        sorted_words = dict(sorted(meaningful_words.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        logger.info(f"Found {len(sorted_words)} meaningful words")
        return sorted_words

    def find_collocations(self, tokens: List[str], top_n: int = 20) -> Dict[str, List[Tuple]]:
        """
        Find bigram and trigram collocations.
        
        Args:
            tokens: Preprocessed tokens
            top_n: Number of top collocations to return
            
        Returns:
            Dictionary with bigrams and trigrams
        """
        logger.info("Finding collocations...")
        
        collocations = {}
        
        # Bigram collocations
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigram_finder.apply_freq_filter(3)  # Must appear at least 3 times
        bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
        collocations['bigrams'] = bigrams
        
        # Trigram collocations
        trigram_finder = TrigramCollocationFinder.from_words(tokens)
        trigram_finder.apply_freq_filter(2)  # Must appear at least 2 times
        trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, top_n)
        collocations['trigrams'] = trigrams
        
        logger.info(f"Found {len(bigrams)} bigrams and {len(trigrams)} trigrams")
        return collocations

    def extract_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract relations using POS tagging and named entity recognition.
        
        Args:
            text: Original text
            
        Returns:
            Dictionary containing various relation types
        """
        logger.info("Extracting relations...")
        
        sentences = sent_tokenize(text)
        relations = {
            'named_entities': [],
            'medical_terms': [],
            'drug_disease_relations': [],
            'treatment_relations': []
        }
        
        # Medical term patterns
        medical_patterns = [
            r'\b\w*sepsis\w*\b', r'\b\w*shock\w*\b', r'\b\w*infection\w*\b',
            r'\b\w*therapy\w*\b', r'\b\w*treatment\w*\b', r'\b\w*antibiotic\w*\b',
            r'\b\w*antimicrobial\w*\b', r'\b\w*mortality\w*\b', r'\b\w*patient\w*\b'
        ]
        
        for sentence in sentences[:100]:  # Limit for performance
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Named Entity Recognition
            try:
                chunks = ne_chunk(pos_tags)
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        relations['named_entities'].append({
                            'text': entity_text,
                            'label': chunk.label(),
                            'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence
                        })
            except:
                pass  # Skip if NER fails
            
            # Extract medical terms
            sentence_lower = sentence.lower()
            for pattern in medical_patterns:
                matches = re.findall(pattern, sentence_lower)
                for match in matches:
                    if len(match) > 3:
                        relations['medical_terms'].append({
                            'term': match,
                            'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence
                        })
        
        logger.info(f"Extracted {len(relations['named_entities'])} named entities and {len(relations['medical_terms'])} medical terms")
        return relations

    def calculate_metrics(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Calculate various text metrics.
        
        Args:
            text: Original text
            tokens: Preprocessed tokens
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating text metrics...")
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Basic metrics
        metrics = {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'meaningful_words': len(tokens),
            'unique_words': len(set(tokens)),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0
        }
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        metrics['sentiment'] = sentiment_scores
        
        # Readability approximation (Flesch-like)
        avg_sentence_length = metrics['avg_sentence_length']
        avg_syllables_per_word = 1.5  # Approximation for medical text
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        metrics['readability_score'] = max(0, min(100, flesch_score))
        
        logger.info("Text metrics calculated")
        return metrics

    def create_visualizations(self, meaningful_words: Dict[str, int], 
                            collocations: Dict[str, List[Tuple]], 
                            metrics: Dict[str, Any],
                            relations: Dict[str, Any]):
        """
        Create various visualization graphs.
        
        Args:
            meaningful_words: Word frequencies
            collocations: Bigrams and trigrams
            metrics: Text metrics
            relations: Extracted relations
        """
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Word frequency bar chart
        plt.figure(figsize=(12, 8))
        words = list(meaningful_words.keys())[:20]
        freqs = list(meaningful_words.values())[:20]
        
        plt.subplot(2, 2, 1)
        plt.barh(words, freqs)
        plt.title('Top 20 Most Frequent Words', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency')
        plt.gca().invert_yaxis()
        
        # 2. Word cloud
        plt.subplot(2, 2, 2)
        if meaningful_words:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(meaningful_words)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Word Cloud', fontsize=14, fontweight='bold')
            plt.axis('off')
        
        # 3. Metrics visualization
        plt.subplot(2, 2, 3)
        metric_names = ['Lexical Diversity', 'Readability Score/100', 'Sentiment Compound']
        metric_values = [
            metrics['lexical_diversity'],
            metrics['readability_score'] / 100,
            (metrics['sentiment']['compound'] + 1) / 2  # Normalize to 0-1
        ]
        bars = plt.bar(metric_names, metric_values)
        plt.title('Text Quality Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score (0-1)')
        plt.xticks(rotation=45)
        
        # Color bars based on values
        for bar, value in zip(bars, metric_values):
            if value > 0.7:
                bar.set_color('green')
            elif value > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 4. Collocation network (simple)
        plt.subplot(2, 2, 4)
        if collocations['bigrams']:
            G = nx.Graph()
            for bigram in collocations['bigrams'][:10]:
                G.add_edge(bigram[0], bigram[1])
            
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   node_size=1000, font_size=8, font_weight='bold')
            plt.title('Bigram Collocations Network', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'nlp_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Detailed bigram chart
        plt.figure(figsize=(10, 6))
        if collocations['bigrams']:
            bigram_strs = [' '.join(bg) for bg in collocations['bigrams'][:15]]
            bigram_counts = range(len(bigram_strs), 0, -1)  # Reverse ranking
            
            plt.barh(bigram_strs, bigram_counts)
            plt.title('Top Bigram Collocations', fontsize=16, fontweight='bold')
            plt.xlabel('Ranking (Higher = More Significant)')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bigram_collocations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Medical terms frequency
        if relations['medical_terms']:
            plt.figure(figsize=(10, 6))
            term_counts = Counter([term['term'] for term in relations['medical_terms']])
            top_terms = dict(term_counts.most_common(15))
            
            plt.bar(top_terms.keys(), top_terms.values())
            plt.title('Medical Terms Frequency', fontsize=16, fontweight='bold')
            plt.xlabel('Medical Terms')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'medical_terms_frequency.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Visualizations created and saved")

    def save_results(self, meaningful_words: Dict[str, int], 
                    collocations: Dict[str, List[Tuple]], 
                    metrics: Dict[str, Any],
                    relations: Dict[str, Any]):
        """
        Save analysis results to files.
        
        Args:
            meaningful_words: Word frequencies
            collocations: Bigrams and trigrams
            metrics: Text metrics
            relations: Extracted relations
        """
        logger.info("Saving analysis results...")
        
        # Save to JSON
        results = {
            'meaningful_words': meaningful_words,
            'collocations': {
                'bigrams': [list(bg) for bg in collocations['bigrams']],
                'trigrams': [list(tg) for tg in collocations['trigrams']]
            },
            'metrics': metrics,
            'relations': relations
        }
        
        with open(self.output_dir / 'nlp_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save to CSV for easy viewing
        pd.DataFrame(list(meaningful_words.items()), 
                    columns=['Word', 'Frequency']).to_csv(
                        self.output_dir / 'meaningful_words.csv', index=False)
        
        # Save metrics summary
        with open(self.output_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=== Medical Text NLP Analysis Summary ===\n\n")
            f.write(f"Total characters: {metrics['total_characters']:,}\n")
            f.write(f"Total words: {metrics['total_words']:,}\n")
            f.write(f"Total sentences: {metrics['total_sentences']:,}\n")
            f.write(f"Meaningful words: {metrics['meaningful_words']:,}\n")
            f.write(f"Unique words: {metrics['unique_words']:,}\n")
            f.write(f"Lexical diversity: {metrics['lexical_diversity']:.3f}\n")
            f.write(f"Average sentence length: {metrics['avg_sentence_length']:.1f} words\n")
            f.write(f"Average word length: {metrics['avg_word_length']:.1f} characters\n")
            f.write(f"Readability score: {metrics['readability_score']:.1f}\n")
            f.write(f"Sentiment (compound): {metrics['sentiment']['compound']:.3f}\n\n")
            
            f.write("Top 10 meaningful words:\n")
            for i, (word, freq) in enumerate(list(meaningful_words.items())[:10], 1):
                f.write(f"{i:2d}. {word}: {freq}\n")
            
            f.write(f"\nTop 10 bigrams:\n")
            for i, bigram in enumerate(collocations['bigrams'][:10], 1):
                f.write(f"{i:2d}. {' '.join(bigram)}\n")
        
        logger.info("Results saved to files")

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Perform complete NLP analysis on a medical document.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting NLP analysis of: {file_path}")
        
        # Read the document
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Clean the text
        cleaned_text = self.clean_markdown_text(raw_text)
        
        # Preprocess
        tokens = self.preprocess_text(cleaned_text)
        
        # Perform analysis
        meaningful_words = self.extract_meaningful_words(tokens)
        collocations = self.find_collocations(tokens)
        relations = self.extract_relations(cleaned_text)
        metrics = self.calculate_metrics(cleaned_text, tokens)
        
        # Create visualizations
        self.create_visualizations(meaningful_words, collocations, metrics, relations)
        
        # Save results
        self.save_results(meaningful_words, collocations, metrics, relations)
        
        logger.info("NLP analysis completed successfully")
        
        return {
            'meaningful_words': meaningful_words,
            'collocations': collocations,
            'relations': relations,
            'metrics': metrics
        }


def main():
    """
    Main function to run the NLP analysis on the test document.
    """
    # Initialize analyzer with output directory in project root
    project_root = Path(__file__).parent.parent.parent  # Go up from NLP/V2/ to project root
    output_dir = project_root / "nlp_output"
    analyzer = MedicalNLPAnalyzer(output_dir=str(output_dir))
    
    # Test document path (absolute path to ensure it works from any directory)
    test_file = project_root / "output" / "docling_md" / "jcm-12-03188_enhanced.md"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        print(f"Error: Test file not found: {test_file}")
        return
    
    # Run analysis
    results = analyzer.analyze_document(str(test_file))
    
    # Print summary
    print("\n=== NLP Analysis Completed ===")
    print(f"Output directory: {output_dir}/")
    print(f"Meaningful words found: {len(results['meaningful_words'])}")
    print(f"Bigrams found: {len(results['collocations']['bigrams'])}")
    print(f"Trigrams found: {len(results['collocations']['trigrams'])}")
    print(f"Named entities found: {len(results['relations']['named_entities'])}")
    print(f"Medical terms found: {len(results['relations']['medical_terms'])}")
    print(f"Lexical diversity: {results['metrics']['lexical_diversity']:.3f}")
    print(f"Readability score: {results['metrics']['readability_score']:.1f}")
    print(f"Sentiment: {results['metrics']['sentiment']['compound']:.3f}")
    
    print("\nTop 10 meaningful words:")
    for i, (word, freq) in enumerate(list(results['meaningful_words'].items())[:10], 1):
        print(f"{i:2d}. {word}: {freq}")
    
    print("\nFiles generated in nlp_output/:")
    print("- nlp_analysis_overview.png (main visualizations)")
    print("- bigram_collocations.png (detailed bigram chart)")
    print("- medical_terms_frequency.png (medical terms frequency)")
    print("- nlp_analysis_results.json (complete results)")
    print("- meaningful_words.csv (word frequencies)")
    print("- analysis_summary.txt (text summary)")


if __name__ == "__main__":
    main()