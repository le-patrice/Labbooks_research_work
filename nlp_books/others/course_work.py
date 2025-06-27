
"""
Literary Language Evolution Analysis System
Analyzes linguistic changes across novels spanning different time periods
"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams

# PDF processing
import PyPDF2

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# # Download required NLTK data
# nltk_downloads = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 
#                   'words', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4']
# for item in nltk_downloads:
#     try:
#         nltk.data.find(f'tokenizers/{item}')
#     except LookupError:
#         nltk.download(item, quiet=True)

class TextProcessor:
    """Handles text extraction and preprocessing"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def read_txt_file(self, filepath: str) -> str:
        """Read text from .txt file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as file:
                return file.read()
    
    def read_pdf_file(self, filepath: str) -> str:
        """Read text from .pdf file"""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\']', ' ', text)
        return text.strip()

class LinguisticAnalyzer:
    """Main linguistic analysis class"""
    
    def __init__(self):
        self.processor = TextProcessor()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.novels_data = {
            "Bartleby, the Scrivener": {"year": 1853, "path": None},
            "The Yellow Wallpaper": {"year": 1892, "path": None},
            "Passing": {"year": 1929, "path": None},
            "A Small Place": {"year": 1988, "path": None},
            "We Should All Be Feminists": {"year": 2014, "path": None}
        }
        
        # Define keyword categories
        self.keywords = {
            "identity": ["woman", "self", "identity", "individual", "person", "character"],
            "morality_sin": ["sin", "moral", "virtue", "evil", "good", "right", "wrong"],
            "technology_nature": ["machine", "nature", "natural", "artificial", "modern", "progress"],
            "disappearance_absence": ["absent", "disappear", "vanish", "gone", "lost", "empty"]
        }
        
        self.results = {}
    
    def extract_linguistic_features(self, text: str, title: str) -> Dict[str, Any]:
        """Extract comprehensive linguistic features from text"""
        # Clean text
        clean_text = self.processor.clean_text(text)
        
        # Tokenization
        sentences = sent_tokenize(clean_text)
        words = word_tokenize(clean_text.lower())
        words_clean = [word for word in words if word.isalpha() and word not in self.processor.stop_words]
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # Named Entity Recognition
        entities = []
        try:
            chunks = ne_chunk(pos_tags)
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entities.append((' '.join([token for token, pos in chunk.leaves()]), chunk.label()))
        except:
            pass
        
        # Lemmatization and Stemming
        lemmas = [self.processor.lemmatizer.lemmatize(word) for word in words_clean]
        stems = [self.processor.stemmer.stem(word) for word in words_clean]
        
        # N-grams
        bigrams = list(ngrams(words_clean, 2))
        trigrams = list(ngrams(words_clean, 3))
        
        # Tense analysis (simplified)
        tense_patterns = {
            'past': ['VBD', 'VBN'],
            'present': ['VBP', 'VBZ', 'VBG'],
            'future': []  # Will be identified by context
        }
        
        tense_counts = defaultdict(int)
        for word, pos in pos_tags:
            for tense, pos_list in tense_patterns.items():
                if pos in pos_list:
                    tense_counts[tense] += 1
        
        # Frequency analysis
        word_freq = Counter(words_clean)
        lemma_freq = Counter(lemmas)
        pos_freq = Counter([pos for word, pos in pos_tags])
        
        return {
            'title': title,
            'word_count': len(words_clean),
            'sentence_count': len(sentences),
            'unique_words': len(set(words_clean)),
            'words': words_clean,
            'lemmas': lemmas,
            'stems': stems,
            'pos_tags': pos_tags,
            'entities': entities,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'word_freq': word_freq,
            'lemma_freq': lemma_freq,
            'pos_freq': pos_freq,
            'tense_counts': dict(tense_counts)
        }
    
    def create_linguistic_table(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Create detailed linguistic features table"""
        data = []
        for i, (word, pos) in enumerate(features['pos_tags'][:1000]):  # Limit for performance
            if word.isalpha():
                lemma = self.processor.lemmatizer.lemmatize(word.lower())
                stem = self.processor.stemmer.stem(word.lower())
                
                # Simple tense identification
                tense = 'unknown'
                if pos in ['VBD', 'VBN']:
                    tense = 'past'
                elif pos in ['VBP', 'VBZ', 'VBG']:
                    tense = 'present'
                
                # Simple morpheme analysis (suffix-based)
                morphemes = self.analyze_morphemes(word)
                
                data.append({
                    'word': word.lower(),
                    'lemma': lemma,
                    'stem': stem,
                    'pos_tag': pos,
                    'tense': tense,
                    'morphemes': morphemes
                })
        
        return pd.DataFrame(data)
    
    def analyze_morphemes(self, word: str) -> str:
        """Simple morpheme analysis based on common suffixes"""
        suffixes = ['-ing', '-ed', '-er', '-est', '-ly', '-tion', '-sion', '-ness', '-ment']
        morphemes = [word]
        
        for suffix in suffixes:
            if word.endswith(suffix[1:]):
                root = word[:-len(suffix[1:])]
                if len(root) > 2:  # Ensure meaningful root
                    morphemes = [root, suffix]
                    break
        
        return '+'.join(morphemes)
    
    def sentiment_analysis(self, text: str, words: List[str]) -> Dict[str, float]:
        """Perform sentiment analysis using VADER"""
        # Overall text sentiment
        overall_sentiment = self.vader_analyzer.polarity_scores(text)
        
        # Keyword-specific sentiment
        keyword_sentiments = {}
        sentences = sent_tokenize(text)
        
        for category, keyword_list in self.keywords.items():
            sentiments = []
            for sentence in sentences:
                for keyword in keyword_list:
                    if keyword in sentence.lower():
                        sent_score = self.vader_analyzer.polarity_scores(sentence)
                        sentiments.append(sent_score['compound'])
            
            if sentiments:
                keyword_sentiments[category] = np.mean(sentiments)
            else:
                keyword_sentiments[category] = 0.0
        
        return {
            'overall': overall_sentiment,
            'keyword_categories': keyword_sentiments
        }
    
    def calculate_tfidf(self, documents: List[str], titles: List[str]) -> pd.DataFrame:
        """Calculate TF-IDF scores for major words identification"""
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        # Create DataFrame with TF-IDF scores
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=titles
        )
        
        return tfidf_df
    
    def calculate_itf(self, word_frequencies: Dict[str, Counter], titles: List[str]) -> pd.DataFrame:
        """Calculate Inverse Term Frequency for time period characterization"""
        # Combine all word frequencies
        all_words = set()
        for freq_dict in word_frequencies.values():
            all_words.update(freq_dict.keys())
        
        # Calculate document frequency for each word
        doc_freq = {}
        for word in all_words:
            doc_freq[word] = sum(1 for freq_dict in word_frequencies.values() if word in freq_dict)
        
        # Calculate ITF (Inverse Term Frequency)
        itf_data = {}
        total_docs = len(word_frequencies)
        
        for title, freq_dict in word_frequencies.items():
            itf_scores = {}
            for word in all_words:
                tf = freq_dict.get(word, 0)
                if tf > 0:
                    itf = tf * np.log(total_docs / doc_freq[word])
                    itf_scores[word] = itf
                else:
                    itf_scores[word] = 0
            itf_data[title] = itf_scores
        
        return pd.DataFrame(itf_data).fillna(0)
    
    def track_keyword_evolution(self) -> pd.DataFrame:
        """Track keyword frequency evolution across time periods"""
        evolution_data = []
        
        for title, data in self.results.items():
            year = self.novels_data[title]['year']
            word_freq = data['linguistic_features']['word_freq']
            
            for category, keywords in self.keywords.items():
                for keyword in keywords:
                    frequency = word_freq.get(keyword, 0)
                    evolution_data.append({
                        'title': title,
                        'year': year,
                        'category': category,
                        'keyword': keyword,
                        'frequency': frequency,
                        'normalized_freq': frequency / data['linguistic_features']['word_count'] * 1000
                    })
        
        return pd.DataFrame(evolution_data)
    
    def visualize_keyword_trends(self, keyword: str, save_path: str = None):
        """Create combined line and bar graph for keyword usage trends"""
        evolution_df = self.track_keyword_evolution()
        keyword_data = evolution_df[evolution_df['keyword'] == keyword]
        
        if keyword_data.empty:
            print(f"No data found for keyword: {keyword}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Line graph: frequency across years
        yearly_freq = keyword_data.groupby('year')['normalized_freq'].sum().reset_index()
        ax1.plot(yearly_freq['year'], yearly_freq['normalized_freq'], 
                marker='o', linewidth=2, markersize=8)
        ax1.set_title(f'Evolution of "{keyword}" Usage Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Normalized Frequency (per 1000 words)')
        ax1.grid(True, alpha=0.3)
        
        # Bar graph: usage in different contexts (by novel)
        ax2.bar(keyword_data['title'], keyword_data['normalized_freq'], 
               color=sns.color_palette("husl", len(keyword_data)))
        ax2.set_title(f'"{keyword}" Usage by Novel', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Novel')
        ax2.set_ylabel('Normalized Frequency (per 1000 words)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_user_defined_word(self, word: str) -> Dict[str, Any]:
        """Comprehensive analysis for user-defined word"""
        analysis = {
            'word': word,
            'usage_summary': {},
            'semantic_contexts': [],
            'syntactic_patterns': {},
            'sentiment_evolution': {},
            'collocations': {}
        }
        
        for title, data in self.results.items():
            year = self.novels_data[title]['year']
            features = data['linguistic_features']
            
            # Usage frequency
            freq = features['word_freq'].get(word, 0)
            normalized_freq = freq / features['word_count'] * 1000
            
            analysis['usage_summary'][title] = {
                'year': year,
                'frequency': freq,
                'normalized_frequency': normalized_freq
            }
            
            # Semantic contexts (sentences containing the word)
            text = data['text']
            sentences = sent_tokenize(text)
            contexts = [sent for sent in sentences if word in sent.lower()]
            analysis['semantic_contexts'].extend([(title, year, context) for context in contexts[:3]])
            
            # Syntactic patterns (POS tags around the word)
            pos_contexts = []
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(words)
            
            for i, (w, pos) in enumerate(pos_tags):
                if w == word:
                    context_start = max(0, i-2)
                    context_end = min(len(pos_tags), i+3)
                    context_pos = [pos for _, pos in pos_tags[context_start:context_end]]
                    pos_contexts.append(context_pos)
            
            analysis['syntactic_patterns'][title] = pos_contexts
            
            # Sentiment around the word
            word_sentences = [sent for sent in sentences if word in sent.lower()]
            if word_sentences:
                sentiments = [self.vader_analyzer.polarity_scores(sent)['compound'] 
                             for sent in word_sentences]
                analysis['sentiment_evolution'][title] = {
                    'year': year,
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments)
                }
        
        return analysis
    
    def generate_word_analysis_report(self, word: str) -> str:
        """Generate comprehensive report for a word analysis"""
        analysis = self.analyze_user_defined_word(word)
        
        report = f"=== COMPREHENSIVE ANALYSIS: '{word.upper()}' ===\n\n"
        
        # Usage Summary
        report += "1. USAGE FREQUENCY EVOLUTION:\n"
        report += "-" * 40 + "\n"
        for title, data in sorted(analysis['usage_summary'].items(), 
                                 key=lambda x: x[1]['year']):
            report += f"{data['year']} - {title}:\n"
            report += f"  • Raw frequency: {data['frequency']}\n"
            report += f"  • Normalized (per 1000 words): {data['normalized_frequency']:.2f}\n\n"
        
        # Semantic Evolution
        report += "2. SEMANTIC CONTEXTS ACROSS TIME:\n"
        report += "-" * 40 + "\n"
        for title, year, context in sorted(analysis['semantic_contexts'], 
                                          key=lambda x: x[1]):
            report += f"{year} - {title}:\n"
            report += f"  \"{context[:100]}...\"\n\n"
        
        # Sentiment Evolution
        report += "3. SENTIMENT EVOLUTION:\n"
        report += "-" * 40 + "\n"
        for title, data in sorted(analysis['sentiment_evolution'].items(), 
                                 key=lambda x: x[1]['year']):
            sentiment_label = "Positive" if data['avg_sentiment'] > 0.1 else \
                             "Negative" if data['avg_sentiment'] < -0.1 else "Neutral"
            report += f"{data['year']} - {title}: {sentiment_label} "
            report += f"(Score: {data['avg_sentiment']:.3f})\n"
        
        return report
    
    def run_full_analysis(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """Run complete linguistic analysis pipeline"""
        # Update file paths
        for title, path in file_paths.items():
            if title in self.novels_data:
                self.novels_data[title]['path'] = path
        
        print("Starting comprehensive linguistic analysis...")
        
        documents = []
        titles = []
        word_frequencies = {}
        
        # Process each novel
        for title, info in self.novels_data.items():
            if info['path'] is None:
                print(f"Warning: No file path provided for '{title}'")
                continue
            
            print(f"Processing: {title} ({info['year']})")
            
            # Read text
            if info['path'].endswith('.pdf'):
                text = self.processor.read_pdf_file(info['path'])
            else:
                text = self.processor.read_txt_file(info['path'])
            
            if not text.strip():
                print(f"Warning: No text extracted from {title}")
                continue
            
            # Extract linguistic features
            features = self.extract_linguistic_features(text, title)
            
            # Create linguistic table
            ling_table = self.create_linguistic_table(features)
            
            # Sentiment analysis
            sentiment = self.sentiment_analysis(text, features['words'])
            
            # Store results
            self.results[title] = {
                'text': text,
                'linguistic_features': features,
                'linguistic_table': ling_table,
                'sentiment': sentiment
            }
            
            documents.append(text)
            titles.append(title)
            word_frequencies[title] = features['word_freq']
        
        if not documents:
            print("Error: No documents were successfully processed")
            return {}
        
        # Calculate TF-IDF
        print("Calculating TF-IDF scores...")
        tfidf_df = self.calculate_tfidf(documents, titles)
        
        # Calculate ITF
        print("Calculating Inverse Term Frequency...")
        itf_df = self.calculate_itf(word_frequencies, titles)
        
        # Track keyword evolution
        print("Tracking keyword evolution...")
        evolution_df = self.track_keyword_evolution()
        
        # Compile final results
        final_results = {
            'individual_analyses': self.results,
            'tfidf_scores': tfidf_df,
            'itf_scores': itf_df,
            'keyword_evolution': evolution_df,
            'novels_data': self.novels_data
        }
        
        print("Analysis complete!")
        return final_results

# Convenience functions for easy usage
def analyze_word_evolution(analyzer: LinguisticAnalyzer, word: str):
    """Convenience function to analyze a specific word"""
    print(f"\n{'='*60}")
    print(f"ANALYZING WORD: '{word.upper()}'")
    print('='*60)
    
    # Generate report
    report = analyzer.generate_word_analysis_report(word)
    print(report)
    
    # Create visualization
    analyzer.visualize_keyword_trends(word)
    
    return analyzer.analyze_user_defined_word(word)

def main_analysis_pipeline(file_paths: Dict[str, str]) -> LinguisticAnalyzer:
    """Main analysis pipeline - easy entry point"""
    analyzer = LinguisticAnalyzer()
    results = analyzer.run_full_analysis(file_paths)
    
    if results:
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        # Display keyword evolution trends
        evolution_df = results['keyword_evolution']
        print("\nTop evolving keywords:")
        top_keywords = evolution_df.groupby('keyword')['normalized_freq'].sum().sort_values(ascending=False).head(10)
        for keyword, total_freq in top_keywords.items():
            print(f"  • {keyword}: {total_freq:.2f} (total normalized frequency)")
        
        # Display TF-IDF insights
        tfidf_df = results['tfidf_scores']
        print(f"\nTF-IDF Analysis complete - {len(tfidf_df.columns)} features analyzed")
        print("Top distinguishing terms per novel:")
        for novel in tfidf_df.index:
            top_terms = tfidf_df.loc[novel].sort_values(ascending=False).head(3)
            terms_str = ", ".join([f"{term}({score:.3f})" for term, score in top_terms.items()])
            print(f"  • {novel}: {terms_str}")
    
    return analyzer

# Example usage
if __name__ == "__main__":
    # Example file paths - update these with actual file locations
    example_paths = {
        "Bartleby, The Scrivener": "books/bartleby_1853.txt",
        "We Should All Be Feminists": "books/feminist_2014.txt",
        "Passing":"books/nella_1929.txt",
        "A Small Place":"books/small_place_1988.txt",
        "The Yellow Wallpaper":"books/yellow_1892.txt"

    }
    
    print("Literary Language Evolution Analysis System")
    print("=" * 50)
    print("This system analyzes linguistic changes across novels from different time periods.")
    print("Please ensure your text files are available at the specified paths.")
    print("\nTo use this system:")
    print("1. Update the file_paths dictionary with your actual file locations")
    print("2. Run: analyzer = main_analysis_pipeline(file_paths)")
    print("3. Analyze specific words: analyze_word_evolution(analyzer, 'your_word')")
    print("4. Create custom visualizations using analyzer.visualize_keyword_trends('word')")
    
    # Uncomment to run with actual files:
    analyzer = main_analysis_pipeline(example_paths)
    analyze_word_evolution(analyzer, "newspaper")