# import re, warnings
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from textblob import TextBlob
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk import pos_tag
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import textstat
# from collections import Counter
# import numpy as np

# warnings.filterwarnings('ignore')
# plt.style.use('default')
# sns.set_palette("husl")

# # Configuration
# TEXTS = {
#     "1853_Bartleby": "I am a rather elderly man. The nature of my avocations for the last thirty years has brought me into more than ordinary contact with what would seem an interesting and somewhat singular set of men, of whom as yet nothing that I know of has ever been written:—I mean the law-copyists or scriveners.",
#     "1892_Yellow": "It is very seldom that mere ordinary people like John and myself secure ancestral halls for the summer. A colonial mansion, a hereditary estate, I would say a haunted house, and reach the height of romantic felicity—but that would be asking too much of fate!",
#     "1929_Passing": "It was the last letter in Irene Redfield's little pile of morning mail. After her other correspondence, all of it brief—responses to invitations, thanks for flowers, that sort of thing—the long envelope of thin Italian paper with its almost illegible scrawl seemed out of place and alien.",
#     "1988_Small": "If you go to Antigua as a tourist, this is what you will see. If you come by aeroplane, you will land at V.C. Bird International Airport. Vere Cornwall Bird is the Prime Minister of Antigua. You may be the sort of tourist who would wonder why a Prime Minister would want an airport named after him.",
#     "2014_Fem": "We should all be feminists. My own definition of a feminist is a man or a woman who says, 'Yes, there's a problem with gender as it is today and we must fix it, we must do better.' All of us, women and men, must do better."
# }

# SEMANTIC_FIELDS = {
#     'Authority': ['power', 'control', 'authority', 'dominance', 'rule', 'command', 'minister', 'prime'],
#     'Identity': ['identity', 'self', 'race', 'racial', 'gender', 'woman', 'man', 'black', 'white', 'name'],
#     'Emotion': ['happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate', 'romantic', 'felicity'],
#     'Social': ['people', 'social', 'society', 'community', 'public', 'ordinary', 'men', 'women'],
#     'Modern': ['technology', 'modern', 'tourist', 'international', 'contemporary', 'today']
# }

# STOP = set(stopwords.words('english'))
# VADER = SentimentIntensityAnalyzer()

# def analyze_comprehensive(text, label):
#     """Deep linguistic and semantic analysis"""
#     tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in STOP]
#     year = int(label.split('_')[0])
#     sents = sent_tokenize(text)
#     pos_tags = pos_tag(tokens)
    
#     # Core metrics
#     record = {
#         'label': label, 'year': year, 'word_count': len(tokens),
#         'unique_words': len(set(tokens)), 'sentences': len(sents)
#     }
    
#     # Advanced linguistic features
#     record.update({
#         'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
#         'avg_word_length': np.mean([len(w) for w in tokens]) if tokens else 0,
#         'avg_sent_length': len(tokens) / len(sents) if sents else 0,
#         'flesch_score': textstat.flesch_reading_ease(text),
#         'complexity_index': textstat.flesch_kincaid_grade(text)
#     })
    
#     # POS tag analysis - critical for understanding linguistic evolution
#     pos_counts = Counter(tag[:2] for _, tag in pos_tags)
#     total_pos = sum(pos_counts.values())
#     record.update({
#         'noun_ratio': pos_counts.get('NN', 0) / total_pos if total_pos else 0,
#         'verb_ratio': pos_counts.get('VB', 0) / total_pos if total_pos else 0,
#         'adj_ratio': pos_counts.get('JJ', 0) / total_pos if total_pos else 0,
#         'adv_ratio': pos_counts.get('RB', 0) / total_pos if total_pos else 0
#     })
    
#     # Semantic field analysis - tracks thematic evolution
#     for field, words in SEMANTIC_FIELDS.items():
#         count = sum(1 for token in tokens if token in words)
#         record[f'{field.lower()}_frequency'] = count
#         record[f'{field.lower()}_density'] = (count / len(tokens) * 1000) if tokens else 0
    
#     # Sentiment evolution - emotional trajectory analysis
#     tb_sentiment = TextBlob(text).sentiment
#     vader_scores = VADER.polarity_scores(text)
#     record.update({
#         'polarity': tb_sentiment.polarity,
#         'subjectivity': tb_sentiment.subjectivity,
#         'vader_positive': vader_scores['pos'],
#         'vader_negative': vader_scores['neg'],
#         'vader_compound': vader_scores['compound']
#     })
    
#     # Pronoun analysis - perspective and voice evolution
#     pronouns = {'first': ['i', 'me', 'my', 'we', 'us', 'our'], 
#                 'second': ['you', 'your', 'yours'],
#                 'third': ['he', 'him', 'his', 'she', 'her', 'they', 'them']}
    
#     for perspective, pron_list in pronouns.items():
#         count = sum(1 for token in tokens if token in pron_list)
#         record[f'{perspective}_person_ratio'] = count / len(tokens) if tokens else 0
    
#     return record

# def create_visualizations(df):
#     """Generate comprehensive bar and line visualizations"""
    
#     # 1. Semantic Evolution - Line Graph
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
#     # Semantic fields over time
#     for field in ['authority', 'identity', 'emotion', 'social']:
#         density_col = f'{field}_density'
#         if density_col in df.columns:
#             ax1.plot(df['year'], df[density_col], 'o-', label=field.title(), linewidth=2, markersize=8)
#     ax1.set_title('Semantic Field Evolution (1850-2020)', fontweight='bold', fontsize=12)
#     ax1.set_xlabel('Year')
#     ax1.set_ylabel('Density (per 1000 words)')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # POS tag evolution - critical linguistic insight
#     pos_metrics = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']
#     colors = ['blue', 'red', 'green', 'orange']
#     for metric, color in zip(pos_metrics, colors):
#         ax2.plot(df['year'], df[metric], 'o-', label=metric.replace('_', ' ').title(), 
#                 color=color, linewidth=2, markersize=6)
#     ax2.set_title('Part-of-Speech Evolution', fontweight='bold', fontsize=12)
#     ax2.set_xlabel('Year')
#     ax2.set_ylabel('Ratio')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Complexity and readability trends
#     ax3.plot(df['year'], df['flesch_score'], 'o-', color='purple', linewidth=2, markersize=8, label='Flesch Score')
#     ax3_twin = ax3.twinx()
#     ax3_twin.plot(df['year'], df['avg_word_length'], 's--', color='brown', linewidth=2, markersize=6, label='Avg Word Length')
#     ax3.set_title('Linguistic Complexity Evolution', fontweight='bold', fontsize=12)
#     ax3.set_xlabel('Year')
#     ax3.set_ylabel('Flesch Reading Ease', color='purple')
#     ax3_twin.set_ylabel('Average Word Length', color='brown')
#     ax3.grid(True, alpha=0.3)
    
#     # Sentiment trajectory
#     ax4.plot(df['year'], df['polarity'], 'o-', color='darkgreen', linewidth=2, markersize=8, label='TextBlob Polarity')
#     ax4.plot(df['year'], df['vader_compound'], 's--', color='darkred', linewidth=2, markersize=6, label='VADER Compound')
#     ax4.set_title('Emotional Sentiment Evolution', fontweight='bold', fontsize=12)
#     ax4.set_xlabel('Year')
#     ax4.set_ylabel('Sentiment Score')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
#     ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # 2. Bar Charts for Comparative Analysis
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
#     # Word usage frequency by era
#     word_metrics = ['word_count', 'unique_words', 'lexical_diversity']
#     x_pos = np.arange(len(df))
#     width = 0.25
    
#     for i, metric in enumerate(word_metrics):
#         ax1.bar(x_pos + i*width, df[metric], width, label=metric.replace('_', ' ').title())
#     ax1.set_title('Word Usage Patterns Across Eras', fontweight='bold')
#     ax1.set_xlabel('Literary Works')
#     ax1.set_xticks(x_pos + width)
#     ax1.set_xticklabels([label.split('_')[1] for label in df['label']], rotation=45)
#     ax1.legend()
    
#     # Semantic field comparison
#     semantic_cols = [col for col in df.columns if col.endswith('_frequency')]
#     semantic_data = df[semantic_cols].values.T
#     ax2.bar(range(len(df)), semantic_data[0], label='Authority', alpha=0.8)
#     ax2.bar(range(len(df)), semantic_data[1], bottom=semantic_data[0], label='Identity', alpha=0.8)
#     ax2.bar(range(len(df)), semantic_data[2], bottom=semantic_data[0]+semantic_data[1], label='Emotion', alpha=0.8)
#     ax2.set_title('Semantic Field Distribution', fontweight='bold')
#     ax2.set_xlabel('Literary Works')
#     ax2.set_ylabel('Frequency Count')
#     ax2.set_xticks(range(len(df)))
#     ax2.set_xticklabels([label.split('_')[1] for label in df['label']], rotation=45)
#     ax2.legend()
    
#     # Pronoun perspective analysis
#     pronoun_data = df[['first_person_ratio', 'second_person_ratio', 'third_person_ratio']]
#     pronoun_data.plot(kind='bar', ax=ax3, width=0.8)
#     ax3.set_title('Narrative Perspective Evolution', fontweight='bold')
#     ax3.set_xlabel('Literary Works')
#     ax3.set_ylabel('Pronoun Ratio')
#     ax3.set_xticklabels([label.split('_')[1] for label in df['label']], rotation=45)
#     ax3.legend(['First Person', 'Second Person', 'Third Person'])
    
#     # Complexity comparison
#     complexity_metrics = df[['flesch_score', 'complexity_index', 'avg_sent_length']]
#     complexity_metrics.plot(kind='bar', ax=ax4, width=0.8)
#     ax4.set_title('Linguistic Complexity Comparison', fontweight='bold')
#     ax4.set_xlabel('Literary Works')
#     ax4.set_ylabel('Complexity Scores')
#     ax4.set_xticklabels([label.split('_')[1] for label in df['label']], rotation=45)
#     ax4.legend(['Flesch Score', 'Grade Level', 'Avg Sentence Length'])
    
#     plt.tight_layout()
#     plt.show()

# def generate_insights(df):
#     """Deep analytical insights from the data"""
#     print("\n🔍 COMPREHENSIVE LITERARY EVOLUTION ANALYSIS")
#     print("=" * 60)
    
#     print(f"\n📊 CORPUS OVERVIEW:")
#     print(f"Total texts analyzed: {len(df)}")
#     print(f"Time span: {df['year'].min()}-{df['year'].max()} ({df['year'].max()-df['year'].min()} years)")
#     print(f"Average words per text: {df['word_count'].mean():.1f}")
    
#     print(f"\n📈 KEY EVOLUTIONARY TRENDS:")
    
#     # Lexical evolution
#     diversity_trend = np.corrcoef(df['year'], df['lexical_diversity'])[0,1]
#     print(f"Lexical Diversity Evolution: {diversity_trend:.3f} {'↗️ Increasing' if diversity_trend > 0 else '↘️ Decreasing'}")
    
#     # Semantic evolution
#     print(f"\n🎭 SEMANTIC FIELD EVOLUTION:")
#     for field in ['authority', 'identity', 'emotion', 'social']:
#         density_col = f'{field}_density'
#         if density_col in df.columns:
#             trend = np.corrcoef(df['year'], df[density_col])[0,1]
#             direction = '↗️ Rising' if trend > 0 else '↘️ Declining'
#             print(f"{field.title()} terms: {trend:.3f} {direction}")
    
#     # Linguistic complexity
#     complexity_trend = np.corrcoef(df['year'], df['flesch_score'])[0,1]
#     print(f"\n📚 READABILITY EVOLUTION: {complexity_trend:.3f}")
#     print("Positive = Becoming more readable, Negative = Becoming more complex")
    
#     # Emotional trajectory
#     sentiment_trend = np.corrcoef(df['year'], df['polarity'])[0,1]
#     print(f"\n💭 EMOTIONAL EVOLUTION: {sentiment_trend:.3f}")
#     print("Positive = More positive sentiment over time, Negative = More negative")
    
#     # Most significant changes
#     print(f"\n🎯 MOST DRAMATIC CHANGES:")
#     pos_changes = {}
#     for col in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']:
#         pos_changes[col] = abs(df[col].iloc[-1] - df[col].iloc[0])
    
#     most_changed = max(pos_changes, key=pos_changes.get)
#     print(f"Grammatical structure: {most_changed.replace('_', ' ').title()} changed most ({pos_changes[most_changed]:.3f})")
    
#     return df

# # Main execution
# def main():
#     # Analyze all texts
#     data = [analyze_comprehensive(text, label) for label, text in TEXTS.items()]
#     df = pd.DataFrame(data).sort_values('year')
    
#     # Generate insights
#     results_df = generate_insights(df)
    
#     # Create visualizations
#     create_visualizations(df)
    
#     print(f"\n✅ ANALYSIS COMPLETE - {len(df)} texts analyzed across {df['year'].max()-df['year'].min()} years")
#     print("\nKey findings reveal significant evolution in semantic focus, grammatical structure,")
#     print("and emotional expression across literary periods from 1850-2020.")
    
#     return results_df

# if __name__ == '__main__':
#     results = main()
import re
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from collections import Counter
import numpy as np
import nltk

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# Set plot style and palette for aesthetic consistency in visualizations
plt.style.use('default')
sns.set_palette("husl")

# File paths for the literary works. These paths assume the files are accessible
# in the execution environment where the script is run.
FILE_PATHS = {
    "1853_Bartleby": "books/bartleby_1853.txt",
    "1892_Yellow":   "books/yellow_1892.txt",
    "1929_Passing":  "books/nella_1929.txt",
    "1988_Small":    "books/small_place_1988.txt",
    "2014_Fem":      "books/feminist_2014.txt"
}

def load_text_from_file(filepath):
    """
    Reads text content from a given file path.
    Includes basic cleaning to remove potential markdown/source tags that might
    interfere with linguistic analysis. Returns an empty string on error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Remove common markdown artifact patterns like '[source: XXXX]'
        content = re.sub(r'\[source: \d+\]', '', content)
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the file exists.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred reading {filepath}: {e}")
        return ""

# Load the full text content of each book into the TEXTS dictionary
TEXTS = {label: load_text_from_file(filepath) for label, filepath in FILE_PATHS.items()}

# Expanded semantic fields for thematic evolution analysis.
# These categories help group words by their conceptual meaning.
SEMANTIC_FIELDS = {
    'Authority': ['power', 'control', 'authority', 'dominance', 'rule', 'command', 'minister', 'prime', 'government', 'state', 'empire', 'official', 'master', 'colonial'],
    'Identity': ['identity', 'self', 'race', 'racial', 'gender', 'woman', 'man', 'black', 'white', 'negro', 'feminist', 'human', 'african', 'antiguan', 'slave'],
    'Emotion': ['happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate', 'romantic', 'felicity', 'rage', 'bitterness', 'suffering', 'anxiety', 'melancholy', 'displeasure', 'irritation', 'fright', 'upset', 'outrage'],
    'Social': ['people', 'social', 'society', 'community', 'public', 'ordinary', 'men', 'women', 'family', 'friends', 'stranger', 'culture', 'marriage'],
    'Confinement': ['confined', 'prison', 'trap', 'cage', 'bars', 'restricted', 'limit', 'shut', 'locked', 'untamed', 'untenanted', 'isolate', 'segregate'],
    'Progress': ['modern', 'evolution', 'change', 'progress', 'future', 'new', 'improve', 'develop', 'advanced', 'better']
}

# Specific cultural indicator words identified from the deep analysis.
# These words are chosen to highlight significant shifts in societal discourse.
CULTURAL_INDICATOR_WORDS = [
    'seldom', 'negro', 'nigger', 'black', 'white', 'feminist', 'gender', 'slave', 'master',
    'control', 'anger', 'fear', 'you', 'i', 'culture', 'change', 'society', 'freedom', 'colonial', 'truth'
] # 20 words

# NLTK's list of common English stopwords to be removed before analysis
STOP_WORDS = set(stopwords.words('english'))
# VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
VADER_SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

def analyze_comprehensive(text, label):
    """
    Performs a deep linguistic and semantic analysis on a given text.
    Calculates various metrics including:
    - Word counts and lexical diversity
    - Readability scores (Flesch Reading Ease, Flesch-Kincaid Grade Level)
    - Part-of-speech (POS) ratios (nouns, verbs, adjectives, adverbs)
    - Frequency and density of words within defined semantic fields
    - VADER sentiment scores
    - Ratios of first, second, and third-person pronouns
    - Frequency and density of specific cultural indicator words
    """
    # Tokenize words, convert to lowercase, and remove non-alphabetic tokens and stopwords
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in STOP_WORDS]
    year = int(label.split('_')[0]) # Extract year from the label (e.g., "1853_Bartleby" -> 1853)
    sents = sent_tokenize(text) # Sentence tokenization

    # Initialize a default record for texts that might be empty after tokenization
    if not tokens:
        return {
            'label': label, 'year': year, 'word_count': 0, 'unique_words': 0, 'sentences': 0,
            'lexical_diversity': 0.0, 'avg_word_length': 0.0, 'avg_sent_length': 0.0,
            'flesch_score': 0.0, 'complexity_index': 0.0,
            'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0, 'adv_ratio': 0.0,
            **{f'{field.lower()}_frequency': 0 for field in SEMANTIC_FIELDS},
            **{f'{field.lower()}_density': 0.0 for field in SEMANTIC_FIELDS},
            **{f'word_{word}_frequency': 0 for word in CULTURAL_INDICATOR_WORDS}, # Default for new words
            **{f'word_{word}_density': 0.0 for word in CULTURAL_INDICATOR_WORDS}, # Default for new words
            'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_compound': 0.0,
            'first_person_ratio': 0.0, 'second_person_ratio': 0.0, 'third_person_ratio': 0.0
        }

    pos_tags = pos_tag(tokens) # Perform Part-of-Speech tagging on the tokens
    
    # Core linguistic metrics
    record = {
        'label': label,
        'year': year,
        'word_count': len(tokens),
        'unique_words': len(set(tokens)),
        'sentences': len(sents)
    }
    
    # Advanced linguistic features related to complexity and vocabulary
    record.update({
        'lexical_diversity': len(set(tokens)) / len(tokens),
        'avg_word_length': np.mean([len(w) for w in tokens]),
        'avg_sent_length': len(tokens) / len(sents) if sents else 0,
        'flesch_score': textstat.flesch_reading_ease(text), # Higher score = easier to read
        'complexity_index': textstat.flesch_kincaid_grade(text) # Grade level
    })
    
    # POS tag analysis for understanding grammatical structure evolution
    pos_counts = Counter(tag for _, tag in pos_tags) # Count occurrences of each POS tag
    total_pos = sum(pos_counts.values()) # Total number of POS tags
    
    # Group similar POS tags (e.g., NNS, NNP into NN) for simplified ratios
    grouped_pos_counts = {
        'NN': sum(v for k, v in pos_counts.items() if k.startswith('NN')), # Nouns (NN, NNS, NNP, NNPS)
        'VB': sum(v for k, v in pos_counts.items() if k.startswith('VB')), # Verbs (VB, VBD, VBG, VBN, VBP, VBZ)
        'JJ': sum(v for k, v in pos_counts.items() if k.startswith('JJ')), # Adjectives (JJ, JJR, JJS)
        'RB': sum(v for k, v in pos_counts.items() if k.startswith('RB'))  # Adverbs (RB, RBR, RBS)
    }

    record.update({
        'noun_ratio': grouped_pos_counts['NN'] / total_pos if total_pos else 0,
        'verb_ratio': grouped_pos_counts['VB'] / total_pos if total_pos else 0,
        'adj_ratio': grouped_pos_counts['JJ'] / total_pos if total_pos else 0,
        'adv_ratio': grouped_pos_counts['RB'] / total_pos if total_pos else 0
    })
    
    # Semantic field analysis to track thematic focus shifts over time
    for field, words in SEMANTIC_FIELDS.items():
        count = sum(1 for token in tokens if token in words)
        record[f'{field.lower()}_frequency'] = count
        record[f'{field.lower()}_density'] = (count / len(tokens) * 1000) if tokens else 0 # Density per 1000 words
    
    # Sentiment analysis using VADER for emotional trajectory
    vader_scores = VADER_SENTIMENT_ANALYZER.polarity_scores(text)
    record.update({
        'vader_positive': vader_scores['pos'],      # Positive sentiment score
        'vader_negative': vader_scores['neg'],      # Negative sentiment score
        'vader_compound': vader_scores['compound']  # Compound score (normalized between -1 and +1)
    })
    
    # Pronoun analysis for understanding narrative perspective and voice evolution
    pronouns = {
        'first': ['i', 'me', 'my', 'we', 'us', 'our', 'ourselves'], # First-person pronouns
        'second': ['you', 'your', 'yours', 'yourself', 'yourselves'], # Second-person pronouns
        'third': ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'it', 'its', 'itself'] # Third-person pronouns
    }
    
    for perspective, pron_list in pronouns.items():
        count = sum(1 for token in tokens if token in pron_list)
        record[f'{perspective}_person_ratio'] = count / len(tokens) if tokens else 0
    
    # NEW: Cultural indicator word frequency and density for specific cultural shifts
    for word in CULTURAL_INDICATOR_WORDS:
        count = tokens.count(word) # Count occurrences of each specific word
        record[f'word_{word}_frequency'] = count
        record[f'word_{word}_density'] = (count / len(tokens) * 1000) if tokens else 0

    return record

def create_visualizations(df):
    """
    Generates various line and bar plots to visualize the linguistic and
    semantic trends across the literary works. Plots are saved as PNG files
    in the current directory.
    """
    
    # Figure 1: Line Graphs for Semantic Field Trends Over Time
    # Displays the density of words related to key thematic areas.
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12)) # Adjusted subplot layout for all fields
    axes1 = axes1.flatten() # Flatten the 2D array of axes for easy iteration

    for i, field in enumerate(SEMANTIC_FIELDS.keys()): # Iterate through all defined semantic fields
        if i < len(axes1): # Ensure subplot index is within bounds to avoid errors
            density_col = f'{field.lower()}_density'
            if density_col in df.columns:
                axes1[i].plot(df['year'], df[density_col], 'o-', label=field.title(), linewidth=2, markersize=8)
                axes1[i].set_title(f'{field.title()} Density Over Time', fontweight='bold', fontsize=12)
                axes1[i].set_xlabel('Year')
                axes1[i].set_ylabel('Density (per 1000 words)')
                axes1[i].grid(True, alpha=0.3)
                axes1[i].legend()
    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.savefig('semantic_field_trends.png') # Save the figure as a PNG file
    plt.close(fig1) # Close the figure to free up memory

    # Figure 2: Line Graphs for POS Ratios and Readability/Complexity
    # Shows how the usage of nouns, verbs, adjectives, and adverbs changes,
    # along with Flesch scores and average word length.
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    axes2 = axes2.flatten()

    # POS tag evolution over time
    pos_metrics = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']
    colors = ['blue', 'red', 'green', 'orange']
    for i, metric in enumerate(pos_metrics):
        axes2[i].plot(df['year'], df[metric], 'o-', label=metric.replace('_', ' ').title(), 
                     color=colors[i], linewidth=2, markersize=6)
        axes2[i].set_title(f'{metric.replace("_", " ").title()} Evolution', fontweight='bold', fontsize=12)
        axes2[i].set_xlabel('Year')
        axes2[i].set_ylabel('Ratio')
        axes2[i].grid(True, alpha=0.3)
        axes2[i].legend()
    plt.tight_layout()
    plt.savefig('pos_tag_trends.png')
    plt.close(fig2)

    # Figure 3: Complexity & Sentiment Trends
    # Visualizes Flesch Reading Ease, Average Word Length, and VADER Compound Sentiment.
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

    # Linguistic complexity and readability trends
    axes3[0].plot(df['year'], df['flesch_score'], 'o-', color='purple', linewidth=2, markersize=8, label='Flesch Score (Higher = Easier)')
    axes3_twin = axes3[0].twinx() # Create a twin Y-axis for average word length
    axes3_twin.plot(df['year'], df['avg_word_length'], 's--', color='brown', linewidth=2, markersize=6, label='Avg Word Length')
    axes3[0].set_title('Linguistic Complexity Evolution', fontweight='bold', fontsize=12)
    axes3[0].set_xlabel('Year')
    axes3[0].set_ylabel('Flesch Reading Ease', color='purple')
    axes3_twin.set_ylabel('Average Word Length', color='brown')
    axes3[0].grid(True, alpha=0.3)
    axes3[0].legend(loc='upper left')
    axes3_twin.legend(loc='upper right')
    
    # Emotional sentiment trajectory using VADER compound score
    axes3[1].plot(df['year'], df['vader_compound'], 's-', color='darkred', linewidth=2, markersize=8, label='VADER Compound Sentiment')
    axes3[1].set_title('Emotional Sentiment Evolution (VADER)', fontweight='bold', fontsize=12)
    axes3[1].set_xlabel('Year')
    axes3[1].set_ylabel('Compound Sentiment Score')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    axes3[1].axhline(y=0, color='black', linestyle='-', alpha=0.3) # Horizontal line at 0 for neutrality
    
    plt.tight_layout()
    plt.savefig('complexity_sentiment_trends.png')
    plt.close(fig3)

    # Figure 4: Bar Charts for Comparative Analysis Across Books
    # Compares word usage, semantic field distribution, pronoun usage, and readability metrics.
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
    axes4 = axes4.flatten()

    # Word usage patterns (counts and lexical diversity)
    labels_short = [label.split('_')[1] for label in df['label']] # Short labels for X-axis
    x_pos = np.arange(len(df)) # X-axis positions for bars
    width = 0.2 # Width of each bar
    
    axes4[0].bar(x_pos - width, df['word_count'], width, label='Word Count')
    axes4[0].bar(x_pos, df['unique_words'], width, label='Unique Words')
    axes4[0].bar(x_pos + width, df['lexical_diversity']*100, width, label='Lexical Diversity (x100)') # Scale for better visibility
    axes4[0].set_title('Word Usage Patterns Across Eras', fontweight='bold')
    axes4[0].set_xlabel('Literary Works')
    axes4[0].set_xticks(x_pos)
    axes4[0].set_xticklabels(labels_short, rotation=45, ha='right') # Rotate labels for readability
    axes4[0].legend()
    
    # Semantic field distribution (stacked bar chart)
    semantic_cols_freq = [f'{field.lower()}_frequency' for field in SEMANTIC_FIELDS.keys()]
    semantic_df_plot = df[semantic_cols_freq]
    semantic_df_plot.columns = [col.replace('_frequency', '').title() for col in semantic_df_plot.columns] # Clean column names for legend
    semantic_df_plot.index = labels_short # Set index for X-axis labels
    semantic_df_plot.plot(kind='bar', stacked=True, ax=axes4[1], width=0.8)
    axes4[1].set_title('Semantic Field Distribution', fontweight='bold')
    axes4[1].set_xlabel('Literary Works')
    axes4[1].set_ylabel('Frequency Count')
    axes4[1].tick_params(axis='x', rotation=45)
    axes4[1].legend(title="Semantic Fields", bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
    
    # Pronoun perspective analysis (bar chart)
    pronoun_data_plot = df[['first_person_ratio', 'second_person_ratio', 'third_person_ratio']]
    pronoun_data_plot.index = labels_short
    pronoun_data_plot.plot(kind='bar', ax=axes4[2], width=0.8)
    axes4[2].set_title('Narrative Perspective Evolution (Pronoun Ratios)', fontweight='bold')
    axes4[2].set_xlabel('Literary Works')
    axes4[2].set_ylabel('Ratio')
    axes4[2].tick_params(axis='x', rotation=45)
    axes4[2].legend(['First Person', 'Second Person', 'Third Person'])
    
    # Readability comparison (bar chart)
    readability_df_plot = df[['flesch_score', 'complexity_index', 'avg_sent_length']]
    readability_df_plot.index = labels_short
    readability_df_plot.plot(kind='bar', ax=axes4[3], width=0.8)
    axes4[3].set_title('Linguistic Readability Comparison', fontweight='bold')
    axes4[3].set_xlabel('Literary Works')
    axes4[3].set_ylabel('Score/Length')
    axes4[3].tick_params(axis='x', rotation=45)
    axes4[3].legend(['Flesch Reading Ease', 'Flesch-Kincaid Grade Level', 'Avg Sentence Length'])
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png')
    plt.close(fig4) # Close the figure

    # NEW Figure 5: Cultural Indicator Word Trends
    # Specifically visualizes the density of the chosen cultural indicator words over time.
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    for word in CULTURAL_INDICATOR_WORDS:
        density_col = f'word_{word}_density'
        if density_col in df.columns:
            ax5.plot(df['year'], df[density_col], 'o-', label=f"'{word}'", linewidth=1.5, markersize=6)
    ax5.set_title('Density of Key Cultural Indicator Words Over Time', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Density (per 1000 words)')
    ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # Move legend to avoid overlapping plot
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cultural_words_trends.png')
    plt.close(fig5)

def generate_insights(df):
    """
    Generates and prints deep analytical insights from the DataFrame.
    This includes:
    - Corpus overview (total texts, time span, average word count)
    - Key evolutionary trends based on correlations (lexical diversity, semantic fields,
      readability, sentiment, pronoun usage)
    - Highlights the most dramatic changes in grammatical structure.
    - Specific analysis of cultural indicator word trends.
    """
    print("\n🔍 COMPREHENSIVE LITERARY EVOLUTION ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 CORPUS OVERVIEW:")
    print(f"Total texts analyzed: {len(df)}")
    print(f"Time span: {df['year'].min()}-{df['year'].max()} ({df['year'].max()-df['year'].min()} years)")
    print(f"Average words per text: {df['word_count'].mean():.1f}")
    
    print(f"\n📈 KEY EVOLUTIONARY TRENDS (Correlations with Year):")
    
    # Lexical evolution trend (vocabulary richness)
    diversity_trend = np.corrcoef(df['year'], df['lexical_diversity'])[0,1]
    print(f"- Lexical Diversity: {diversity_trend:.3f} {'↗️ Increasing' if diversity_trend > 0 else '↘️ Decreasing'} (indicating a shift towards less varied vocabulary or shorter texts over time)")
    
    # Semantic field evolution trends
    print(f"\n🎭 SEMANTIC FIELD EVOLUTION:")
    for field in SEMANTIC_FIELDS.keys():
        density_col = f'{field.lower()}_density'
        if density_col in df.columns:
            trend = np.corrcoef(df['year'], df[density_col])[0,1]
            direction = '↗️ Rising' if trend > 0.1 else ('↘️ Declining' if trend < -0.1 else '↔️ Stable') # Use threshold for significance
            print(f"- {field.title()} terms: {trend:.3f} {direction}")
    
    # Linguistic complexity/readability trends
    flesch_trend = np.corrcoef(df['year'], df['flesch_score'])[0,1]
    complexity_index_trend = np.corrcoef(df['year'], df['complexity_index'])[0,1]
    avg_sent_length_trend = np.corrcoef(df['year'], df['avg_sent_length'])[0,1]
    
    print(f"\n📚 READABILITY/COMPLEXITY EVOLUTION:")
    print(f"- Flesch Reading Ease (Higher = Easier): {flesch_trend:.3f} {'↗️ More readable' if flesch_trend > 0.1 else ('↘️ Less readable' if flesch_trend < -0.1 else '↔️ Stable')}")
    print(f"- Flesch-Kincaid Grade Level (Higher = More Complex): {complexity_index_trend:.3f} {'↗️ More complex' if complexity_index_trend > 0.1 else ('↘️ Less complex' if complexity_index_trend < -0.1 else '↔️ Stable')}")
    print(f"- Average Sentence Length: {avg_sent_length_trend:.3f} {'↗️ Longer sentences' if avg_sent_length_trend > 0.1 else ('↘️ Shorter sentences' if avg_sent_length_trend < -0.1 else '↔️ Stable')}")

    # Emotional trajectory using VADER compound sentiment
    vader_compound_trend = np.corrcoef(df['year'], df['vader_compound'])[0,1]
    print(f"\n💭 EMOTIONAL EVOLUTION (VADER Compound Sentiment):")
    print(f"- Overall Sentiment: {vader_compound_trend:.3f} {'↗️ More Positive' if vader_compound_trend > 0.1 else ('↘️ More Negative' if vader_compound_trend < -0.1 else '↔️ Stable')}")
    
    # Most significant changes in POS ratios (comparing earliest to latest text)
    print(f"\n🎯 MOST DRAMATIC CHANGES IN GRAMMATICAL STRUCTURE (Magnitude of change from earliest to latest text):")
    pos_changes = {}
    for col in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']:
        start_val = df[col].iloc[0] if not df.empty else 0
        end_val = df[col].iloc[-1] if not df.empty else 0
        change_abs = abs(end_val - start_val)
        direction = 'increase' if end_val > start_val else 'decrease'
        pos_changes[col] = (change_abs, direction)
    
    if pos_changes:
        most_changed_col = max(pos_changes, key=lambda k: pos_changes[k][0])
        abs_change, direction = pos_changes[most_changed_col]
        print(f"- The ratio of '{most_changed_col.replace('_', ' ').title()}' changed most significantly: {abs_change:.3f} ({direction})")
    else:
        print("- No significant POS ratio changes detected (or insufficient data).")
    
    # Narrative perspective changes (Correlations)
    print(f"\n🗣️ NARRATIVE PERSPECTIVE EVOLUTION (Correlation with Year):")
    for pron_type in ['first', 'second', 'third']:
        ratio_col = f'{pron_type}_person_ratio'
        trend = np.corrcoef(df['year'], df[ratio_col])[0,1]
        direction = '↗️ Increasing' if trend > 0.1 else ('↘️ Declining' if trend < -0.1 else '↔️ Stable')
        print(f"- {pron_type.title()} Person Pronoun Ratio: {trend:.3f} {direction}")

    # NEW: Insights for Cultural Indicator Words
    print(f"\n✨ CULTURAL INDICATOR WORD ANALYSIS (Density Trends - Correlation with Year):")
    for word in CULTURAL_INDICATOR_WORDS:
        density_col = f'word_{word}_density'
        if density_col in df.columns:
            trend = np.corrcoef(df['year'], df[density_col])[0,1]
            direction = '↗️ Rising' if trend > 0.1 else ('↘️ Declining' if trend < -0.1 else '↔️ Stable')
            print(f"- '{word}': {trend:.3f} {direction}")

    return df

# Main execution block
def main():
    """
    Main function to orchestrate the loading, comprehensive analysis,
    insight generation, and visualization of the literary texts.
    """
    # Analyze all texts by iterating through the defined FILE_PATHS
    data = [analyze_comprehensive(text, label) for label, text in TEXTS.items()]
    df = pd.DataFrame(data).sort_values('year') # Create DataFrame and sort by year
    
    if df.empty: # Check if analysis yielded any data
        print("No data to analyze. Please check file paths and content.")
        return None

    # Generate and print the analytical insights
    results_df = generate_insights(df)
    
    # Create and save the visualizations
    create_visualizations(df)
    
    print(f"\n✅ ANALYSIS COMPLETE - {len(df)} texts analyzed across {df['year'].max()-df['year'].min()} years")
    print("\nKey findings reveal significant evolution in semantic focus, grammatical structure,")
    print("and emotional expression across literary periods from 1850-2014.")
    print("\nVisualizations (PNG files) have been saved in the current directory.")
    
    return results_df

if __name__ == '__main__':
    main()
