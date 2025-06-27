import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.util import ngrams
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------
# Configuration Setup
# ------------------------
BOOK_METADATA = {
    'Tristram Shandy': {'year': 1759, 'themes': ['psychological', 'fragmented', 'narrative']},
    'The Scarlet Letter': {'year': 1850, 'themes': ['religious', 'moral', 'puritan']},
    'Sister Carrie': {'year': 1900, 'themes': ['urban', 'industrial', 'naturalism']},
    'The Martian Chronicles': {'year': 1950, 'themes': ['futuristic', 'colonial', 'technology']},
    'White Teeth': {'year': 2000, 'themes': ['multicultural', 'identity', 'modern']},
    'The Vanishing Half': {'year': 2020, 'themes': ['race', 'identity', 'contemporary']}
}

TARGET_KEYWORDS = [
    'name', 'race', 'self', 'color', 'double', 'sin', 'shame', 'virtue', 'repent',
    'rocket', 'earth', 'mars', 'colonize', 'soil', 'vanish', 'gone', 'half', 
    'missing', 'woman', 'freedom'
]

# ------------------------
# Initialization Functions
# ------------------------
def initialize_nltk():
    """Download required NLTK resources"""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words')
    ]
    
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

def initialize_spacy():
    """Load spaCy model with error handling"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    except OSError:
        try:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        except:
            print("SpaCy model installation failed. Some features will be disabled.")
            nlp = None
    return nlp

# ------------------------
# File Handling
# ------------------------
def read_file(file_path):
    """
    Read text content from files (supports .txt and .pdf)
    
    Args:
        file_path (str): Path to input file
        
    Returns:
        str: Extracted text content
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    elif ext == '.pdf':
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return '\n'.join(text)
        
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ------------------------
# Text Processing
# ------------------------
def preprocess_text(text, stop_words):
    """
    Clean and tokenize text
    
    Args:
        text (str): Raw input text
        stop_words (set): Stopwords to remove
        
    Returns:
        list: Cleaned tokens
    """
    # Normalization
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization and filtering
    tokens = word_tokenize(text)
    return [token for token in tokens 
            if token not in stop_words and len(token) > 2]

# ------------------------
# Linguistic Analysis
# ------------------------
class LinguisticAnalyzer:
    """Core analysis engine for linguistic evolution tracking"""
    
    def __init__(self, book_metadata, target_keywords):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = initialize_spacy()
        self.book_metadata = book_metadata
        self.target_keywords = target_keywords
        self.analyzed_books = {}
        self.tfidf_matrix = None
        self.feature_names = None

    def extract_features(self, text, book_title):
        """
        Extract comprehensive linguistic features from text
        
        Args:
            text (str): Book content
            book_title (str): Book identifier
            
        Returns:
            dict: Extracted features
        """
        features = {
            'title': book_title,
            'year': self.book_metadata[book_title]['year'],
            'word_tokens': [],
            'named_entities': [],
            'pos_tags': [],
            'lemmas': [],
            'stems': [],
            'bigrams': [],
            'trigrams': [],
            'tense_analysis': {},
            'morphological_table': []
        }
        
        # Text preprocessing
        tokens = preprocess_text(text, self.stop_words)
        features['word_tokens'] = tokens
        
        # POS tagging
        features['pos_tags'] = pos_tag(tokens)
        
        # spaCy processing
        if self.nlp:
            doc = self.nlp(' '.join(tokens[:100000]))  # Memory cap
            features['named_entities'] = [(ent.text, ent.label_) for ent in doc.ents]
            features['lemmas'] = [token.lemma_ for token in doc]
        
        # Stemming and morphological analysis
        for word, pos in features['pos_tags']:
            features['stems'].append(self.stemmer.stem(word))
            features['morphological_table'].append({
                'word': word,
                'stem': self.stemmer.stem(word),
                'pos': pos,
                'morphemes': self._analyze_morphemes(word)
            })
        
        # N-grams
        features['bigrams'] = list(ngrams(tokens, 2))
        features['trigrams'] = list(ngrams(tokens, 3))
        
        # Tense analysis
        features['tense_analysis'] = self._analyze_tense(features['pos_tags'])
        
        return features

    def _analyze_morphemes(self, word):
        """Identify morphemes in words"""
        morphemes = []
        if word.endswith('ing'):
            morphemes.append('progressive')
        if word.endswith('ed'):
            morphemes.append('past')
        if word.endswith('s') and len(word) > 3:
            morphemes.append('plural')
        return morphemes

    def _analyze_tense(self, pos_tags):
        """Calculate tense distribution"""
        tense_mapping = {
            'VBD': 'past',
            'VBG': 'present_progressive',
            'VBN': 'past_participle',
            'VBP': 'present',
            'VBZ': 'present'
        }
        return Counter(tense_mapping.get(tag, 'other') for _, tag in pos_tags)

    def analyze_sentiment(self, text):
        """Calculate sentiment scores using VADER"""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }

    def calculate_keyword_frequency(self, features, keywords=None):
        """Measure occurrence of target keywords"""
        keywords = keywords or self.target_keywords
        freq_data = {}
        tokens = features['word_tokens']
        
        for keyword in keywords:
            count = sum(1 for token in tokens if token == keyword)
            freq_data[keyword] = {
                'frequency': count,
                'relative_frequency': count / len(tokens) if tokens else 0,
                'contexts': self._extract_contexts(tokens, keyword)
            }
        return freq_data

    def _extract_contexts(self, tokens, keyword, window=5):
        """Capture keyword usage contexts"""
        contexts = []
        for idx in [i for i, t in enumerate(tokens) if t == keyword]:
            start = max(0, idx - window)
            end = min(len(tokens), idx + window + 1)
            contexts.append(' '.join(tokens[start:end]))
        return contexts[:3]  # Return top 3 contexts

    # ... (Other methods remain similar with optimized operations)
    # [calculate_tfidf, calculate_inverse_term_frequency, analyze_book, 
    #  create_evolution_table, visualize_keyword_evolution, etc.]

# ------------------------
# Main Execution
# ------------------------
if __name__ == "__main__":
    # Initialize resources
    initialize_nltk()
    
    # Configure book paths (example)
    BOOK_PATHS = {
        'Tristram Shandy': 'data/books/tristram.txt',
        'The Scarlet Letter': 'data/books/scarlet.pdf',
        # ... other books
    }
    
    # Initialize analyzer
    analyzer = LinguisticAnalyzer(BOOK_METADATA, TARGET_KEYWORDS)
    
    # Process books
    book_texts = {}
    for title, path in BOOK_PATHS.items():
        try:
            text = read_file(path)
            book_texts[title] = text
            analyzer.analyze_book(text, title)
            print(f"Processed: {title}")
        except Exception as e:
            print(f"Error processing {title}: {str(e)}")
    
    # Analysis pipeline
    tfidf_df = analyzer.calculate_tfidf(book_texts)
    itf_results = analyzer.calculate_inverse_term_frequency(tfidf_df)
    
    # Generate outputs
    evolution_df = analyzer.create_evolution_table()
    analyzer.visualize_keyword_evolution('identity', save_fig=True)
    analyzer.generate_comprehensive_report()
    
    # Save results
    evolution_df.to_csv('linguistic_evolution.csv', index=False)
    print("Analysis complete. Results saved.")