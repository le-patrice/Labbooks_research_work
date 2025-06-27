# import re
# import warnings
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# # from textblob import TextBlob # Removed due to ModuleNotFoundError
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk import pos_tag
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import textstat
# from collections import Counter
# import numpy as np
# import nltk


# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')

# # Set plot style and palette for aesthetic consistency
# plt.style.use('default')
# sns.set_palette("husl")

# # File paths from the user's uploaded files (these paths are theoretical for the environment)
# FILE_PATHS = {
#     "1853_Bartleby": "books/bartleby_1853.txt",
#     "1892_Yellow":   "books/yellow_1892.txt",
#     "1929_Passing":  "books/nella_1929.txt",
#     "1988_Small":    "books/small_place_1988.txt",
#     "2014_Fem":      "books/feminist_2014.txt"
# }

# def load_text_from_file(filepath):
#     """
#     Reads text content from a given file path.
#     Includes basic cleaning to remove potential markdown/source tags.
#     """
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             content = f.read()
#         # Remove common markdown artifact patterns like '[source: XXXX]'
#         content = re.sub(r'\[source: \d+\]', '', content)
#         return content
#     except FileNotFoundError:
#         print(f"Error: File not found at {filepath}. Please ensure the file exists.")
#         return ""
#     except Exception as e:
#         print(f"An error occurred reading {filepath}: {e}")
#         return ""

# # Load full texts using the file paths
# # This dictionary will store the actual text content after loading
# TEXTS = {label: load_text_from_file(filepath) for label, filepath in FILE_PATHS.items()}

# # Expanded semantic fields based on the analysis
# SEMANTIC_FIELDS = {
#     'Authority': ['power', 'control', 'authority', 'dominance', 'rule', 'command', 'minister', 'prime', 'government', 'state', 'empire', 'official', 'master', 'colonel'],
#     'Identity': ['identity', 'self', 'race', 'racial', 'gender', 'woman', 'man', 'black', 'white', 'negro', 'nigger', 'feminist', 'human', 'african', 'antiguan', 'slave', 'master'],
#     'Emotion': ['happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate', 'romantic', 'felicity', 'rage', 'bitterness', 'suffering', 'anxiety', 'melancholy', 'displeasure', 'irritation', 'fright', 'disappointed', 'upset', 'outrage'],
#     'Social': ['people', 'social', 'society', 'community', 'public', 'ordinary', 'men', 'women', 'family', 'friends', 'stranger', 'culture', 'marriage', 'class', 'neighbor'],
#     'Confinement': ['confined', 'prison', 'trap', 'cage', 'bars', 'restricted', 'limit', 'shut', 'locked', 'untamed', 'untenanted', 'isolate', 'segregate', 'confined'],
#     'Progress': ['modern', 'evolution', 'change', 'progress', 'future', 'new', 'improve', 'develop', 'advanced'] # Added to capture shifts over time
# }

# # Initialize NLTK components
# STOP_WORDS = set(stopwords.words('english'))
# VADER_SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

# def analyze_comprehensive(text, label):
#     """
#     Performs deep linguistic and semantic analysis on a given text.
#     Calculates various metrics including word counts, readability scores,
#     part-of-speech ratios, semantic field frequencies, sentiment, and pronoun usage.
#     """
#     # Tokenize and normalize words, filtering out non-alphabetic tokens and stopwords
#     tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in STOP_WORDS]
#     year = int(label.split('_')[0])
#     sents = sent_tokenize(text)
    
#     # Initialize a default record for empty texts to prevent errors
#     if not tokens:
#         return {
#             'label': label, 'year': year, 'word_count': 0, 'unique_words': 0, 'sentences': 0,
#             'lexical_diversity': 0.0, 'avg_word_length': 0.0, 'avg_sent_length': 0.0,
#             'flesch_score': 0.0, 'complexity_index': 0.0,
#             'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0, 'adv_ratio': 0.0,
#             **{f'{field.lower()}_frequency': 0 for field in SEMANTIC_FIELDS},
#             **{f'{field.lower()}_density': 0.0 for field in SEMANTIC_FIELDS},
#             'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_compound': 0.0,
#             'first_person_ratio': 0.0, 'second_person_ratio': 0.0, 'third_person_ratio': 0.0
#         }

#     pos_tags = pos_tag(tokens) # Part-of-speech tagging
    
#     # Core metrics calculation
#     record = {
#         'label': label,
#         'year': year,
#         'word_count': len(tokens),
#         'unique_words': len(set(tokens)),
#         'sentences': len(sents)
#     }
    
#     # Advanced linguistic features
#     record.update({
#         'lexical_diversity': len(set(tokens)) / len(tokens),
#         'avg_word_length': np.mean([len(w) for w in tokens]),
#         'avg_sent_length': len(tokens) / len(sents) if sents else 0, # Handle zero sentences
#         'flesch_score': textstat.flesch_reading_ease(text),
#         'complexity_index': textstat.flesch_kincaid_grade(text)
#     })
    
#     # POS tag analysis for grammatical structure evolution
#     pos_counts = Counter(tag for _, tag in pos_tags)
#     total_pos = sum(pos_counts.values())
    
#     # Group common POS tags for higher-level analysis
#     grouped_pos_counts = {
#         'NN': sum(v for k, v in pos_counts.items() if k.startswith('NN')), # Nouns (singular, plural, proper)
#         'VB': sum(v for k, v in pos_counts.items() if k.startswith('VB')), # Verbs (all tenses)
#         'JJ': sum(v for k, v in pos_counts.items() if k.startswith('JJ')), # Adjectives
#         'RB': sum(v for k, v in pos_counts.items() if k.startswith('RB'))  # Adverbs
#     }

#     record.update({
#         'noun_ratio': grouped_pos_counts['NN'] / total_pos if total_pos else 0,
#         'verb_ratio': grouped_pos_counts['VB'] / total_pos if total_pos else 0,
#         'adj_ratio': grouped_pos_counts['JJ'] / total_pos if total_pos else 0,
#         'adv_ratio': grouped_pos_counts['RB'] / total_pos if total_pos else 0
#     })
    
#     # Semantic field analysis to track thematic focus shifts
#     for field, words in SEMANTIC_FIELDS.items():
#         count = sum(1 for token in tokens if token in words)
#         record[f'{field.lower()}_frequency'] = count
#         record[f'{field.lower()}_density'] = (count / len(tokens) * 1000) if tokens else 0 # Density per 1000 words
    
#     # Sentiment analysis using VADER (TextBlob removed)
#     vader_scores = VADER_SENTIMENT_ANALYZER.polarity_scores(text)
#     record.update({
#         'vader_positive': vader_scores['pos'],
#         'vader_negative': vader_scores['neg'],
#         'vader_compound': vader_scores['compound'] # Compound score provides overall sentiment
#     })
    
#     # Pronoun analysis for narrative perspective changes
#     pronouns = {
#         'first': ['i', 'me', 'my', 'we', 'us', 'our', 'ourselves'], 
#         'second': ['you', 'your', 'yours', 'yourself', 'yourselves'],
#         'third': ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'it', 'its', 'itself']
#     }
    
#     for perspective, pron_list in pronouns.items():
#         count = sum(1 for token in tokens if token in pron_list)
#         record[f'{perspective}_person_ratio'] = count / len(tokens) if tokens else 0
    
#     return record

# def create_visualizations(df):
#     """
#     Generates various line and bar plots to visualize the linguistic and
#     semantic trends across the literary works. Plots are saved as PNG files.
#     """
    
#     # Line Graphs for Trends Over Time (Semantic Fields)
#     fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12)) # Adjusted for more semantic fields
#     axes1 = axes1.flatten()

#     for i, field in enumerate(SEMANTIC_FIELDS.keys()): # Iterate through all semantic fields
#         if i < len(axes1): # Ensure subplot index is within bounds
#             density_col = f'{field.lower()}_density'
#             if density_col in df.columns:
#                 axes1[i].plot(df['year'], df[density_col], 'o-', label=field.title(), linewidth=2, markersize=8)
#                 axes1[i].set_title(f'{field.title()} Density Over Time', fontweight='bold', fontsize=12)
#                 axes1[i].set_xlabel('Year')
#                 axes1[i].set_ylabel('Density (per 1000 words)')
#                 axes1[i].grid(True, alpha=0.3)
#                 axes1[i].legend()
#     plt.tight_layout()
#     plt.savefig('semantic_field_trends.png')
#     plt.close(fig1) # Close the figure to free memory

#     # Line Graphs for Trends Over Time (POS Ratios and Complexity/Sentiment)
#     fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
#     axes2 = axes2.flatten()

#     # POS tag evolution
#     pos_metrics = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']
#     colors = ['blue', 'red', 'green', 'orange']
#     for i, metric in enumerate(pos_metrics):
#         axes2[i].plot(df['year'], df[metric], 'o-', label=metric.replace('_', ' ').title(), 
#                      color=colors[i], linewidth=2, markersize=6)
#         axes2[i].set_title(f'{metric.replace("_", " ").title()} Evolution', fontweight='bold', fontsize=12)
#         axes2[i].set_xlabel('Year')
#         axes2[i].set_ylabel('Ratio')
#         axes2[i].grid(True, alpha=0.3)
#         axes2[i].legend()
#     plt.tight_layout()
#     plt.savefig('pos_tag_trends.png')
#     plt.close(fig2)

#     fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

#     # Complexity and readability trends
#     axes3[0].plot(df['year'], df['flesch_score'], 'o-', color='purple', linewidth=2, markersize=8, label='Flesch Score (Higher = Easier)')
#     axes3_twin = axes3[0].twinx()
#     axes3_twin.plot(df['year'], df['avg_word_length'], 's--', color='brown', linewidth=2, markersize=6, label='Avg Word Length')
#     axes3[0].set_title('Linguistic Complexity Evolution', fontweight='bold', fontsize=12)
#     axes3[0].set_xlabel('Year')
#     axes3[0].set_ylabel('Flesch Reading Ease', color='purple')
#     axes3_twin.set_ylabel('Average Word Length', color='brown')
#     axes3[0].grid(True, alpha=0.3)
#     axes3[0].legend(loc='upper left')
#     axes3_twin.legend(loc='upper right')
    
#     # Sentiment trajectory (VADER compound score)
#     axes3[1].plot(df['year'], df['vader_compound'], 's-', color='darkred', linewidth=2, markersize=8, label='VADER Compound Sentiment')
#     axes3[1].set_title('Emotional Sentiment Evolution (VADER)', fontweight='bold', fontsize=12)
#     axes3[1].set_xlabel('Year')
#     axes3[1].set_ylabel('Compound Sentiment Score')
#     axes3[1].legend()
#     axes3[1].grid(True, alpha=0.3)
#     axes3[1].axhline(y=0, color='black', linestyle='-', alpha=0.3) # Add zero line for sentiment
    
#     plt.tight_layout()
#     plt.savefig('complexity_sentiment_trends.png')
#     plt.close(fig3)

#     # Bar Charts for Comparative Analysis
#     fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
#     axes4 = axes4.flatten()

#     # Word usage frequency by era
#     labels_short = [label.split('_')[1] for label in df['label']]
#     x_pos = np.arange(len(df))
#     width = 0.2
    
#     axes4[0].bar(x_pos - width, df['word_count'], width, label='Word Count')
#     axes4[0].bar(x_pos, df['unique_words'], width, label='Unique Words')
#     axes4[0].bar(x_pos + width, df['lexical_diversity']*100, width, label='Lexical Diversity (x100)') # Scale for better visibility
#     axes4[0].set_title('Word Usage Patterns Across Eras', fontweight='bold')
#     axes4[0].set_xlabel('Literary Works')
#     axes4[0].set_xticks(x_pos)
#     axes4[0].set_xticklabels(labels_short, rotation=45, ha='right')
#     axes4[0].legend()
    
#     # Semantic field comparison (stacked bar chart)
#     semantic_cols_freq = [f'{field.lower()}_frequency' for field in SEMANTIC_FIELDS.keys()]
#     semantic_df_plot = df[semantic_cols_freq]
#     semantic_df_plot.columns = [col.replace('_frequency', '').title() for col in semantic_df_plot.columns]
#     semantic_df_plot.index = labels_short
#     semantic_df_plot.plot(kind='bar', stacked=True, ax=axes4[1], width=0.8)
#     axes4[1].set_title('Semantic Field Distribution', fontweight='bold')
#     axes4[1].set_xlabel('Literary Works')
#     axes4[1].set_ylabel('Frequency Count')
#     axes4[1].tick_params(axis='x', rotation=45)
#     axes4[1].legend(title="Semantic Fields", bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside to prevent overlap
    
#     # Pronoun perspective analysis
#     pronoun_data_plot = df[['first_person_ratio', 'second_person_ratio', 'third_person_ratio']]
#     pronoun_data_plot.index = labels_short
#     pronoun_data_plot.plot(kind='bar', ax=axes4[2], width=0.8)
#     axes4[2].set_title('Narrative Perspective Evolution (Pronoun Ratios)', fontweight='bold')
#     axes4[2].set_xlabel('Literary Works')
#     axes4[2].set_ylabel('Ratio')
#     axes4[2].tick_params(axis='x', rotation=45)
#     axes4[2].legend(['First Person', 'Second Person', 'Third Person'])
    
#     # Readability comparison (Flesch Score, Kincaid Grade, Avg Sentence Length)
#     readability_df_plot = df[['flesch_score', 'complexity_index', 'avg_sent_length']]
#     readability_df_plot.index = labels_short
#     readability_df_plot.plot(kind='bar', ax=axes4[3], width=0.8)
#     axes4[3].set_title('Linguistic Readability Comparison', fontweight='bold')
#     axes4[3].set_xlabel('Literary Works')
#     axes4[3].set_ylabel('Score/Length')
#     axes4[3].tick_params(axis='x', rotation=45)
#     axes4[3].legend(['Flesch Reading Ease', 'Flesch-Kincaid Grade Level', 'Avg Sentence Length'])
    
#     plt.tight_layout()
#     plt.savefig('comparative_analysis.png')
#     plt.close(fig4) # Close the figure

# def generate_insights(df):
#     """
#     Generates and prints deep analytical insights from the DataFrame.
#     This includes corpus overview, evolutionary trends based on correlations,
#     and highlights significant changes in specific linguistic features.
#     """
#     print("\n🔍 COMPREHENSIVE LITERARY EVOLUTION ANALYSIS")
#     print("=" * 60)
    
#     print(f"\n📊 CORPUS OVERVIEW:")
#     print(f"Total texts analyzed: {len(df)}")
#     print(f"Time span: {df['year'].min()}-{df['year'].max()} ({df['year'].max()-df['year'].min()} years)")
#     print(f"Average words per text: {df['word_count'].mean():.1f}")
    
#     print(f"\n📈 KEY EVOLUTIONARY TRENDS (Correlations with Year):")
    
#     # Lexical evolution trend
#     diversity_trend = np.corrcoef(df['year'], df['lexical_diversity'])[0,1]
#     print(f"- Lexical Diversity: {diversity_trend:.3f} {'↗️ Increasing' if diversity_trend > 0 else '↘️ Decreasing'} (indicating a shift towards less varied vocabulary or shorter texts)")
    
#     # Semantic field evolution trends
#     print(f"\n🎭 SEMANTIC FIELD EVOLUTION:")
#     for field in SEMANTIC_FIELDS.keys():
#         density_col = f'{field.lower()}_density'
#         if density_col in df.columns:
#             trend = np.corrcoef(df['year'], df[density_col])[0,1]
#             direction = '↗️ Rising' if trend > 0.1 else ('↘️ Declining' if trend < -0.1 else '↔️ Stable') # Add threshold for significance
#             print(f"- {field.title()} terms: {trend:.3f} {direction}")
    
#     # Linguistic complexity/readability trends
#     flesch_trend = np.corrcoef(df['year'], df['flesch_score'])[0,1]
#     complexity_index_trend = np.corrcoef(df['year'], df['complexity_index'])[0,1]
#     avg_sent_length_trend = np.corrcoef(df['year'], df['avg_sent_length'])[0,1]
    
#     print(f"\n📚 READABILITY/COMPLEXITY EVOLUTION:")
#     print(f"- Flesch Reading Ease (Higher = Easier): {flesch_trend:.3f} {'↗️ More readable' if flesch_trend > 0.1 else ('↘️ Less readable' if flesch_trend < -0.1 else '↔️ Stable')}")
#     print(f"- Flesch-Kincaid Grade Level (Higher = More Complex): {complexity_index_trend:.3f} {'↗️ More complex' if complexity_index_trend > 0.1 else ('↘️ Less complex' if complexity_index_trend < -0.1 else '↔️ Stable')}")
#     print(f"- Average Sentence Length: {avg_sent_length_trend:.3f} {'↗️ Longer sentences' if avg_sent_length_trend > 0.1 else ('↘️ Shorter sentences' if avg_sent_length_trend < -0.1 else '↔️ Stable')}")

#     # Emotional trajectory (VADER only)
#     vader_compound_trend = np.corrcoef(df['year'], df['vader_compound'])[0,1]
#     print(f"\n💭 EMOTIONAL EVOLUTION (VADER Compound Sentiment):")
#     print(f"- Overall Sentiment: {vader_compound_trend:.3f} {'↗️ More Positive' if vader_compound_trend > 0.1 else ('↘️ More Negative' if vader_compound_trend < -0.1 else '↔️ Stable')}")
    
#     # Most significant changes in POS ratios (comparing earliest to latest)
#     print(f"\n🎯 MOST DRAMATIC CHANGES IN GRAMMATICAL STRUCTURE (Magnitude of change from earliest to latest text):")
#     pos_changes = {}
#     for col in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']:
#         start_val = df[col].iloc[0] if not df.empty else 0
#         end_val = df[col].iloc[-1] if not df.empty else 0
#         change_abs = abs(end_val - start_val)
#         direction = 'increase' if end_val > start_val else 'decrease'
#         pos_changes[col] = (change_abs, direction) # Store both absolute change and direction
    
#     if pos_changes:
#         # Find the POS tag with the largest absolute change
#         most_changed_col = max(pos_changes, key=lambda k: pos_changes[k][0])
#         abs_change, direction = pos_changes[most_changed_col]
#         print(f"- The ratio of '{most_changed_col.replace('_', ' ').title()}' changed most significantly: {abs_change:.3f} ({direction})")
#     else:
#         print("- No significant POS ratio changes detected (or insufficient data).")
    
#     # Narrative perspective changes (Correlations)
#     print(f"\n🗣️ NARRATIVE PERSPECTIVE EVOLUTION (Correlation with Year):")
#     for pron_type in ['first', 'second', 'third']:
#         ratio_col = f'{pron_type}_person_ratio'
#         trend = np.corrcoef(df['year'], df[ratio_col])[0,1]
#         direction = '↗️ Increasing' if trend > 0.1 else ('↘️ Decreasing' if trend < -0.1 else '↔️ Stable')
#         print(f"- {pron_type.title()} Person Pronoun Ratio: {trend:.3f} {direction}")

#     return df

# # Main execution block
# def main():
#     """
#     Main function to orchestrate the loading, analysis, insight generation,
#     and visualization of the literary texts.
#     """
#     # Analyze all texts
#     data = [analyze_comprehensive(text, label) for label, text in TEXTS.items()]
#     df = pd.DataFrame(data).sort_values('year')
    
#     if df.empty:
#         print("No data to analyze. Please check file paths and content.")
#         return None

#     # Generate insights
#     results_df = generate_insights(df)
    
#     # Create visualizations
#     create_visualizations(df)
    
#     print(f"\n✅ ANALYSIS COMPLETE - {len(df)} texts analyzed across {df['year'].max()-df['year'].min()} years")
#     print("\nKey findings reveal significant evolution in semantic focus, grammatical structure,")
#     print("and emotional expression across literary periods from 1850-2014.")
#     print("\nVisualizations (PNG files) have been saved in the current directory.")
    
#     return results_df

# if __name__ == '__main__':
#     main()

import re, warnings, os
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

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Configuration
FILE_PATHS = {
    "1853_Bartleby": "books/bartleby_1853.txt",
    "1892_Yellow": "books/yellow_1892.txt", 
    "1929_Passing": "books/nella_1929.txt",
    "1988_Small": "books/small_place_1988.txt",
    "2014_Fem": "books/feminist_2014.txt"
}

SEMANTIC_FIELDS = {
    'Authority': ['power', 'control', 'authority', 'dominance', 'rule', 'command', 'minister', 'prime', 'government', 'colonial'],
    'Identity': ['identity', 'self', 'race', 'racial', 'gender', 'woman', 'man', 'black', 'white', 'negro', 'feminist', 'human'],
    'Emotion': ['happy', 'sad', 'angry', 'joy', 'fear', 'love', 'hate', 'romantic', 'rage', 'anxiety', 'melancholy'],
    'Social': ['people', 'social', 'society', 'community', 'public', 'ordinary', 'family', 'culture'],
    'Confinement': ['confined', 'prison', 'trap', 'cage', 'restricted', 'limit', 'locked', 'isolate'],
    'Progress': ['modern', 'evolution', 'change', 'progress', 'future', 'new', 'improve', 'develop', 'advanced']
}

CULTURAL_WORDS = ['seldom', 'negro', 'black', 'white', 'feminist', 'gender', 'slave', 'master', 'control', 'anger', 
                 'fear', 'culture', 'change', 'society', 'freedom', 'colonial', 'truth']

STOP_WORDS = set(stopwords.words('english'))
VADER = SentimentIntensityAnalyzer()

def load_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return re.sub(r'\[source: \d+\]', '', f.read())
    except:
        print(f"Error loading {filepath}")
        return ""

def analyze_text(text, label):
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in STOP_WORDS]
    year = int(label.split('_')[0])
    sents = sent_tokenize(text)
    
    if not tokens:
        return {f: 0 for f in ['word_count', 'unique_words', 'sentences', 'lexical_diversity', 'flesch_score', 
                              'noun_ratio', 'verb_ratio', 'adj_ratio', 'vader_compound', 'first_person_ratio']}
    
    pos_tags = pos_tag(tokens)
    pos_counts = Counter(tag for _, tag in pos_tags)
    total_pos = sum(pos_counts.values())
    
    # Core analysis
    record = {
        'label': label, 'year': year, 'word_count': len(tokens), 'unique_words': len(set(tokens)),
        'sentences': len(sents), 'lexical_diversity': len(set(tokens))/len(tokens),
        'avg_word_length': np.mean([len(w) for w in tokens]), 'flesch_score': textstat.flesch_reading_ease(text),
        'complexity_index': textstat.flesch_kincaid_grade(text)
    }
    
    # POS ratios
    pos_groups = {'NN': 'noun', 'VB': 'verb', 'JJ': 'adj', 'RB': 'adv'}
    for prefix, name in pos_groups.items():
        count = sum(v for k, v in pos_counts.items() if k.startswith(prefix))
        record[f'{name}_ratio'] = count / total_pos if total_pos else 0
    
    # Semantic fields
    for field, words in SEMANTIC_FIELDS.items():
        count = sum(1 for token in tokens if token in words)
        record[f'{field.lower()}_density'] = (count / len(tokens) * 1000) if tokens else 0
    
    # Sentiment & pronouns
    vader_score = VADER.polarity_scores(text)
    record['vader_compound'] = vader_score['compound']
    
    pronouns = {'first': ['i', 'me', 'my', 'we', 'us'], 'second': ['you', 'your'], 
                'third': ['he', 'him', 'she', 'her', 'they', 'them', 'it']}
    for p_type, p_list in pronouns.items():
        count = sum(1 for token in tokens if token in p_list)
        record[f'{p_type}_person_ratio'] = count / len(tokens) if tokens else 0
    
    # Cultural words
    for word in CULTURAL_WORDS:
        count = tokens.count(word)
        record[f'word_{word}_density'] = (count / len(tokens) * 1000) if tokens else 0
    
    return record

def create_visualizations(df):
    # Semantic trends
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, field in enumerate(SEMANTIC_FIELDS.keys()):
        if i < len(axes):
            col = f'{field.lower()}_density'
            axes[i].plot(df['year'], df[col], 'o-', linewidth=2, markersize=6)
            axes[i].set_title(f'{field} Density Over Time', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/semantic_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # POS and complexity
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    pos_metrics = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio']
    colors = ['blue', 'red', 'green', 'orange']
    for i, (metric, color) in enumerate(zip(pos_metrics, colors)):
        axes[i//2, i%2].plot(df['year'], df[metric], 'o-', color=color, linewidth=2)
        axes[i//2, i%2].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        axes[i//2, i%2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/pos_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cultural words heatmap
    cultural_cols = [f'word_{w}_density' for w in CULTURAL_WORDS[:10]]  # Top 10 for readability
    cultural_data = df[cultural_cols].T
    cultural_data.columns = [label.split('_')[1] for label in df['label']]
    cultural_data.index = [col.replace('word_', '').replace('_density', '') for col in cultural_data.index]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cultural_data, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Cultural Indicator Words Density Heatmap', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/cultural_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_insights(df):
    print("🔍 COMPREHENSIVE LITERARY EVOLUTION ANALYSIS")
    print("=" * 50)
    print(f"📊 Analyzed {len(df)} texts spanning {df['year'].max()-df['year'].min()} years")
    print(f"📝 Average words per text: {df['word_count'].mean():.0f}")
    
    # Key trends
    trends = {
        'Lexical Diversity': ('lexical_diversity', 'vocabulary richness'),
        'Readability': ('flesch_score', 'text accessibility'),
        'Sentiment': ('vader_compound', 'emotional tone'),
        'First Person Usage': ('first_person_ratio', 'narrative intimacy')
    }
    
    print("\n📈 KEY EVOLUTIONARY TRENDS:")
    for name, (col, desc) in trends.items():
        corr = np.corrcoef(df['year'], df[col])[0,1]
        direction = '↗️ Rising' if corr > 0.1 else ('↘️ Declining' if corr < -0.1 else '↔️ Stable')
        print(f"• {name}: {corr:.3f} {direction} ({desc})")
    
    # Semantic evolution
    print("\n🎭 THEMATIC EVOLUTION:")
    for field in SEMANTIC_FIELDS.keys():
        col = f'{field.lower()}_density'
        corr = np.corrcoef(df['year'], df[col])[0,1]
        direction = '↗️' if corr > 0.1 else ('↘️' if corr < -0.1 else '↔️')
        print(f"• {field}: {corr:.3f} {direction}")
    
    # Most changed cultural words
    print("\n✨ TOP CULTURAL SHIFTS:")
    cultural_changes = {}
    for word in CULTURAL_WORDS:
        col = f'word_{word}_density'
        if col in df.columns:
            change = abs(df[col].iloc[-1] - df[col].iloc[0])
            cultural_changes[word] = change
    
    top_changes = sorted(cultural_changes.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, change in top_changes:
        print(f"• '{word}': {change:.2f} density change")
    
    return df

def main():
    # Load and analyze texts
    texts = {label: load_text(path) for label, path in FILE_PATHS.items()}
    data = [analyze_text(text, label) for label, text in texts.items()]
    df = pd.DataFrame(data).sort_values('year')
    
    if df.empty:
        print("❌ No data to analyze. Check file paths.")
        return None
    
    # Generate insights and visualizations
    results = generate_insights(df)
    create_visualizations(df)
    
    print(f"\n✅ Analysis complete! Visualizations saved to 'visualizations/' directory")
    return results

# Execute analysis
if __name__ == '__main__':
    results_df = main()