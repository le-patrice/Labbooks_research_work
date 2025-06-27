# # # Comprehensive Analysis: Evolution of Language & Culture (1850–2020)
# # # Tools: Python, NLTK, Pandas, Matplotlib, Seaborn, TextBlob, VADER, NRCLex, textstat, spaCy

# # import os, re
# # import nltk
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from textblob import TextBlob
# # from nltk.tokenize import word_tokenize, sent_tokenize
# # from nltk.corpus import stopwords
# # from nltk.probability import FreqDist
# # from nltk import pos_tag, Text
# # from nltk.sentiment.vader import SentimentIntensityAnalyzer
# # from nrclex import NRCLex
# # import textstat
# # import spacy
# # from collections import Counter

# # # Initialize resources
# # nltk.download('punkt'); nltk.download('stopwords')
# # nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon')
# # STOP = set(stopwords.words('english'))
# # VADER = SentimentIntensityAnalyzer()
# # NLP = spacy.load('en_core_web_sm')

# # # Configuration
# # TEXTS = {
# #     "1853_Bartleby": "books/bartleby_1853.txt",
# #     "1892_Yellow":   "books/yellow_1892.txt",
# #     "1929_Passing":  "books/nella_1929.txt",
# #     "1988_Small":    "books/small_place_1988.txt",
# #     "2014_Fem":      "books/feminist_2014.txt"
# # }
# # UK_US_PAIRS = [('honour', 'honor'), ('labour', 'labor'), ('defence', 'defense')]
# # KEY_TERMS = ['freedom', 'race', 'feminism', 'colonial', 'technology']
# # TOPIC_LEXICONS = {
# #     'Globalization': ['global', 'trade', 'capitalism'],
# #     'Identity': ['identity', 'race', 'gender'],
# #     'Environment': ['climate', 'earth', 'pollution']
# # }

# # # Utility functions

# # def load_text(path):
# #     with open(path, encoding='utf-8') as f:
# #         return f.read()

# # def clean_tokens(text):
# #     return [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in STOP]

# # # Morphological analysis

# # def morphological_diversity(text):
# #     doc = NLP(text)
# #     lemmas = [token.lemma_ for token in doc if token.is_alpha]
# #     return len(set(lemmas)) / len(lemmas) if lemmas else 0

# # def morph_tag_distribution(text):
# #     doc = NLP(text)
# #     morphs = [token.morph.to_str() for token in doc if token.is_alpha]
# #     return Counter(morphs)

# # # Core analyses

# # def sentence_stats(text):
# #     sents = sent_tokenize(text)
# #     lengths = [len(word_tokenize(s)) for s in sents]
# #     return sum(lengths) / len(lengths) if lengths else 0

# # def passive_rate(text):
# #     sents = sent_tokenize(text); count=0
# #     for s in sents:
# #         tags = pos_tag(word_tokenize(s))
# #         if any(w.lower() in ('is','are','was','were','be','been','being') and nxt=='VBN'
# #                for (w,_),( _,nxt) in zip(tags,tags[1:])):
# #             count+=1
# #     return count/len(sents) if sents else 0

# # def negation_rate(tokens):
# #     negs=[w for w in tokens if re.fullmatch(r"not|never|no|none|n't",w)]
# #     return len(negs)/len(tokens) if tokens else 0

# # def dialogue_ratio(text):
# #     quotes=re.findall(r'"(.*?)"',text)
# #     qwords=sum(len(q.split()) for q in quotes)
# #     total=len(word_tokenize(text))
# #     return qwords/total if total else 0

# # def pov_ratio(tokens):
# #     first=tokens.count('i')+tokens.count('we')
# #     third=tokens.count('he')+tokens.count('she')+tokens.count('they')
# #     return first/third if third else 0

# # def pos_diversity(tokens, prefix):
# #     tags=pos_tag(tokens)
# #     words=[w for w,t in tags if t.startswith(prefix)]
# #     return len(set(words))/len(words) if words else 0

# # def sentiment_scores(text):
# #     return TextBlob(text).sentiment.polarity, VADER.polarity_scores(text)['compound']

# # def emotion_profile(tokens):
# #     counts=Counter()
# #     for w in tokens:
# #         counts.update(NRCLex(w).raw_emotion_scores)
# #     return counts

# # def named_entities(text):
# #     doc=NLP(text)
# #     return Counter(ent.label_ for ent in doc.ents)

# # def readability(text):
# #     return textstat.flesch_reading_ease(text), textstat.gunning_fog(text)

# # def topic_trends(tokens):
# #     return {t: sum(tokens.count(w) for w in ws) for t,ws in TOPIC_LEXICONS.items()}

# # # Analyze each text

# # def analyze(path, label, data):
# #     text=load_text(path)
# #     tokens=clean_tokens(text)
# #     record={'Label':label,'Tokens':len(tokens),'Unique':len(set(tokens))}

# #     # Linguistic metrics
# #     record['AvgSentLen']=sentence_stats(text)
# #     record['PassiveRate']=passive_rate(text)
# #     record['NegRate']=negation_rate(tokens)
# #     record['DialogueRatio']=dialogue_ratio(text)
# #     record['POVRatio']=pov_ratio(tokens)
# #     record['AdjDiv']=pos_diversity(tokens,'JJ')
# #     record['VerbDiv']=pos_diversity(tokens,'VB')
# #     record['MorphDiv']=morphological_diversity(text)
# #     # Sentiment
# #     tb,vd=sentiment_scores(text)
# #     record['TBPolarity'],record['VADER']=tb,vd
# #     # Readability
# #     fr,fg=readability(text)
# #     record['Flesch'],record['Fog']=fr,fg
# #     # Pronouns
# #     for p in ['i','we','you','he','she','they']:
# #         record[p]=tokens.count(p)
# #     # Emotion
# #     emo=emotion_profile(tokens)
# #     for e in ['joy','anger','fear','sadness']:
# #         record[e]=emo.get(e,0)
# #     # Entities
# #     ne=named_entities(text)
# #     for ent,c in ne.items(): record[f'ENT_{ent}']=c
# #     # Spelling variants
# #     for uk,us in UK_US_PAIRS:
# #         record[uk]=tokens.count(uk)
# #         record[us]=tokens.count(us)
# #     # Key terms & topics
# #     for t in KEY_TERMS: record[t]=tokens.count(t)
# #     record.update(topic_trends(tokens))

# #     # Morphological tag distribution for later plotting
# #     data.append(record)

# # # Plotting

# # def plot_series(df,x,y,title,annotate=False):
# #     plt.figure(figsize=(10,5))
# #     sns.lineplot(data=df,x=x,y=y,marker='o')
# #     plt.title(title)
# #     plt.xticks(rotation=45)
# #     if annotate:
# #         for xi,yi in zip(df[x],df[y]): plt.text(xi, yi*1.05, f'{yi:.2f}', ha='center')
# #     plt.tight_layout();plt.show()

# # def plot_bar(df,x,y,title):
# #     plt.figure(figsize=(10,5))
# #     sns.barplot(data=df,x=x,y=y)
# #     plt.title(title)
# #     plt.xticks(rotation=45)
# #     plt.tight_layout();plt.show()

# # # Execute
# # if __name__=='__main__':
# #     data=[]
# #     for lbl,path in TEXTS.items(): analyze(path,lbl,data)
# #     df=pd.DataFrame(data)
# #     print(df)

# #     # Numeric plots
# #     metrics_line=['AvgSentLen','PassiveRate','NegRate','DialogueRatio','POVRatio','TBPolarity','VADER','Flesch','Fog']
# #     for m in metrics_line: plot_series(df,'Label',m,m.replace('Rate',' Rate').replace('Avg','Average ').replace('TB','TextBlob ').replace('VADER','VADER ').replace('Flesch','Flesch ').replace('Fog','Gunning Fog ').title(),True)
# #     # POS & Morph
# #     plot_bar(df,'Label','AdjDiv','Adjective Diversity')
# #     plot_bar(df,'Label','VerbDiv','Verb Diversity')
# #     plot_bar(df,'Label','MorphDiv','Morphological Diversity')
# #     # Pronouns
# #     pron_df=df.melt(id_vars='Label',value_vars=['i','we','you','he','she','they'],var_name='Pronoun',value_name='Count')
# #     plot_bar(pron_df,'Label','Count','Pronoun Usage')
# #     # Spelling
# #     for uk,us in UK_US_PAIRS:
# #         pair_df=df.melt(id_vars='Label',value_vars=[uk,us],var_name='Variant',value_name='Count')
# #         plot_bar(pair_df,'Label','Count',f'Spelling {uk} vs {us}')
# #     # Topics
# #     for t in TOPIC_LEXICONS: plot_series(df,'Label',t,f'Theme: {t}',True)
# #     # Emotions
# #     for e in ['joy','anger','fear','sadness']: plot_series(df,'Label',e,f'Emotion: {e}',True)
# #     # Morph tags sample from largest text
# #     largest= max(TEXTS.keys(), key=lambda k: df.loc[df['Label']==k,'Tokens'].values[0])
# #     dist=morph_tag_distribution(load_text(TEXTS[largest]))
# #     morph_df=pd.DataFrame(dist.most_common(10),columns=['Morph','Count'])
# #     plot_bar(morph_df,'Morph','Count','Top Morphological Tags')
# #     # Named Entities summary
# #     ent_cols=[c for c in df.columns if c.startswith('ENT_')]
# #     print('\nNamed Entities:')
# #     print(df[['Label']+ent_cols])
# #     # Concordance
# #     print("\nConcordance Examples (2014_Fem):")
# #     txt=Text(word_tokenize(load_text(TEXTS['2014_Fem']).lower()))
# #     for t in KEY_TERMS:
# #         print(f"-- {t} --"); txt.concordance(t,width=60,lines=2)

# # Enhanced Language Evolution Analysis: Semantic & Linguistic Trends (1850-2020)
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

# # Setup
# warnings.filterwarnings('ignore')
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

# # Initialize NLTK resources
# import nltk
# for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'vader_lexicon']:
#     nltk.download(resource, quiet=True)

# STOP = set(stopwords.words('english'))
# VADER = SentimentIntensityAnalyzer()

# # Book file paths for comprehensive analysis
# BOOK_PATHS = {
#     "1853_Bartleby": "books/bartleby_1853.txt",
#     "1892_Yellow":   "books/yellow_1892.txt", 
#     "1929_Passing":  "books/nella_1929.txt",
#     "1988_Small":    "books/small_place_1988.txt",
#     "2014_Fem":      "books/feminist_2014.txt"
# }

# def load_book_text(filepath, encoding='utf-8'):
#     """Load full text from book files with error handling."""
#     encodings = [encoding, 'latin1', 'cp1252', 'iso-8859-1']
    
#     for enc in encodings:
#         try:
#             with open(filepath, 'r', encoding=enc) as f:
#                 text = f.read()
#                 print(f"✓ Loaded {filepath} ({len(text)} characters, encoding: {enc})")
#                 return text
#         except (UnicodeDecodeError, FileNotFoundError) as e:
#             if enc == encodings[-1]:  # Last encoding attempt
#                 print(f"✗ Failed to load {filepath}: {e}")
#                 return ""
#             continue
#     return ""

# def preprocess_book_text(text):
#     """Clean and preprocess full book text for analysis."""
#     # Remove excessive whitespace and normalize
#     text = re.sub(r'\s+', ' ', text.strip())
    
#     # Remove common book metadata patterns
#     text = re.sub(r'Project Gutenberg.*?(?=Chapter|CHAPTER|\n\n)', '', text, flags=re.DOTALL | re.IGNORECASE)
#     text = re.sub(r'End of.*?Project Gutenberg.*', '', text, flags=re.DOTALL | re.IGNORECASE)
#     text = re.sub(r'\*\*\*.*?\*\*\*', '', text, flags=re.DOTALL)
    
#     # Remove chapter headers and page numbers
#     text = re.sub(r'Chapter \d+|CHAPTER \d+', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'Page \d+|\d+\s*)', '', text, flags=re.IGNORECASE)

# # Target word categories for semantic analysis
# SEMANTIC_CATEGORIES = {
#     'Authority': ['power', 'control', 'authority', 'dominance', 'rule', 'command', 'order'],
#     'Identity': ['identity', 'self', 'race', 'racial', 'gender', 'woman', 'man', 'black', 'white'],
#     'Freedom': ['freedom', 'liberty', 'independence', 'autonomy', 'choice', 'rights', 'equality'],
#     'Colonial': ['colonial', 'colonialism', 'empire', 'imperial', 'dominion', 'exploitation'],
#     'Feminism': ['feminism', 'feminist', 'gender', 'equality', 'women', 'patriarchy'],
#     'Technology': ['technology', 'digital', 'internet', 'computer', 'modern', 'innovation']
# }

# # Utility functions
# def clean_tokens(text):
#     tokens = word_tokenize(text.lower())
#     return [w for w in tokens if w.isalpha() and w not in STOP and len(w) > 2]

# def safe_ratio(num, den):
#     return num / den if den > 0 else 0

# # Core analysis functions
# def linguistic_metrics(text, tokens):
#     """Calculate key linguistic features."""
#     sents = sent_tokenize(text)
#     avg_sent_len = safe_ratio(sum(len(word_tokenize(s)) for s in sents), len(sents))
    
#     # Complexity measures
#     flesch = textstat.flesch_reading_ease(text)
#     ttr = safe_ratio(len(set(tokens)), len(tokens))  # Type-token ratio
    
#     # Syntactic features
#     tags = pos_tag(tokens)
#     adj_count = sum(1 for _, tag in tags if tag.startswith('JJ'))
#     verb_count = sum(1 for _, tag in tags if tag.startswith('VB'))
    
#     return {
#         'avg_sent_len': avg_sent_len,
#         'flesch': flesch,
#         'ttr': ttr,
#         'adj_ratio': safe_ratio(adj_count, len(tokens)),
#         'verb_ratio': safe_ratio(verb_count, len(tokens))
#     }

# def semantic_analysis(tokens):
#     """Analyze semantic categories and target words."""
#     results = {}
    
#     # Count words in each semantic category
#     for category, words in SEMANTIC_CATEGORIES.items():
#         count = sum(tokens.count(word) for word in words)
#         results[f'{category.lower()}_count'] = count
#         results[f'{category.lower()}_density'] = safe_ratio(count, len(tokens)) * 1000
    
#     # Semantic clustering - words that co-occur with target terms
#     semantic_clusters = {}
#     for category, target_words in SEMANTIC_CATEGORIES.items():
#         cluster_words = []
#         for i, token in enumerate(tokens):
#             if token in target_words:
#                 # Get context window (5 words before and after)
#                 start = max(0, i-5)
#                 end = min(len(tokens), i+6)
#                 context = tokens[start:end]
#                 cluster_words.extend([w for w in context if w not in target_words])
        
#         if cluster_words:
#             semantic_clusters[category] = Counter(cluster_words).most_common(3)
    
#     results['semantic_clusters'] = semantic_clusters
#     return results

# def sentiment_evolution(text):
#     """Analyze sentiment with TextBlob and VADER."""
#     tb_sentiment = TextBlob(text).sentiment
#     vader_scores = VADER.polarity_scores(text)
    
#     return {
#         'polarity': tb_sentiment.polarity,
#         'subjectivity': tb_sentiment.subjectivity,
#         'vader_compound': vader_scores['compound'],
#         'vader_positive': vader_scores['pos'],
#         'vader_negative': vader_scores['neg']
#     }

# def pronoun_analysis(tokens):
#     """Analyze pronoun usage patterns."""
#     pronouns = {
#         'first_person': ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'],
#         'second_person': ['you', 'your', 'yours'],
#         'third_person': ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their']
#     }
    
#     results = {}
#     for category, pron_list in pronouns.items():
#         count = sum(tokens.count(p) for p in pron_list)
#         results[f'{category}_count'] = count
#         results[f'{category}_ratio'] = safe_ratio(count, len(tokens))
    
#     return results

# # Main analysis function
# def analyze_text(text, label):
#     """Comprehensive text analysis."""
#     tokens = clean_tokens(text)
#     year = int(label.split('_')[0])
    
#     # Base metrics
#     record = {
#         'label': label,
#         'year': year,
#         'tokens': len(tokens),
#         'unique_tokens': len(set(tokens))
#     }
    
#     # Linguistic analysis
#     record.update(linguistic_metrics(text, tokens))
    
#     # Semantic analysis
#     record.update(semantic_analysis(tokens))
    
#     # Sentiment analysis
#     record.update(sentiment_evolution(text))
    
#     # Pronoun analysis
#     record.update(pronoun_analysis(tokens))
    
#     return record

# # Visualization functions
# def plot_semantic_trends(df):
#     """Plot semantic category trends over time."""
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#     axes = axes.flatten()
    
#     categories = ['authority', 'identity', 'freedom', 'colonial', 'feminism', 'technology']
    
#     for i, category in enumerate(categories):
#         density_col = f'{category}_density'
#         if density_col in df.columns:
#             axes[i].plot(df['year'], df[density_col], 'o-', linewidth=2, markersize=6)
#             axes[i].set_title(f'{category.title()} Terms', fontweight='bold')
#             axes[i].set_xlabel('Year')
#             axes[i].set_ylabel('Density (per 1000 words)')
#             axes[i].grid(True, alpha=0.3)
            
#             # Add trend line
#             z = np.polyfit(df['year'], df[density_col], 1)
#             p = np.poly1d(z)
#             axes[i].plot(df['year'], p(df['year']), '--', alpha=0.7, color='red')
    
#     plt.tight_layout()
#     plt.suptitle('Semantic Category Evolution (1850-2020)', fontsize=16, fontweight='bold', y=1.02)
#     plt.show()

# def plot_linguistic_evolution(df):
#     """Plot linguistic complexity over time."""
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
#     # Sentence complexity
#     ax1.plot(df['year'], df['avg_sent_len'], 'o-', color='blue', linewidth=2)
#     ax1.set_title('Average Sentence Length', fontweight='bold')
#     ax1.set_ylabel('Words per sentence')
#     ax1.grid(True, alpha=0.3)
    
#     # Readability
#     ax2.plot(df['year'], df['flesch'], 'o-', color='green', linewidth=2)
#     ax2.set_title('Flesch Reading Ease', fontweight='bold')
#     ax2.set_ylabel('Reading ease score')
#     ax2.grid(True, alpha=0.3)
    
#     # Vocabulary richness
#     ax3.plot(df['year'], df['ttr'], 'o-', color='orange', linewidth=2)
#     ax3.set_title('Type-Token Ratio', fontweight='bold')
#     ax3.set_ylabel('Vocabulary richness')
#     ax3.set_xlabel('Year')
#     ax3.grid(True, alpha=0.3)
    
#     # Sentiment evolution
#     ax4.plot(df['year'], df['polarity'], 'o-', color='purple', linewidth=2, label='Polarity')
#     ax4.plot(df['year'], df['vader_compound'], 's--', color='red', linewidth=2, label='VADER')
#     ax4.set_title('Sentiment Evolution', fontweight='bold')
#     ax4.set_ylabel('Sentiment score')
#     ax4.set_xlabel('Year')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# def plot_pronoun_trends(df):
#     """Plot pronoun usage trends."""
#     plt.figure(figsize=(10, 6))
    
#     pronoun_types = ['first_person_ratio', 'second_person_ratio', 'third_person_ratio']
#     colors = ['blue', 'green', 'red']
#     labels = ['First Person', 'Second Person', 'Third Person']
    
#     for ptype, color, label in zip(pronoun_types, colors, labels):
#         if ptype in df.columns:
#             plt.plot(df['year'], df[ptype], 'o-', color=color, 
#                     linewidth=2, markersize=6, label=label)
    
#     plt.title('Pronoun Usage Evolution', fontsize=14, fontweight='bold')
#     plt.xlabel('Year', fontweight='bold')
#     plt.ylabel('Ratio (pronouns/total words)', fontweight='bold')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# # Main execution
# def main():
#     print("🔍 Language Evolution Analysis: Semantic & Linguistic Trends")
#     print("=" * 60)
    
#     # Analyze all texts
#     data = []
#     for label, text in TEXTS.items():
#         print(f"📖 Processing {label}...")
#         record = analyze_text(text, label)
#         data.append(record)
    
#     # Create DataFrame and sort by year
#     df = pd.DataFrame(data).sort_values('year')
    
#     # Display key results
#     print("\n📊 Key Metrics Summary:")
#     display_cols = ['label', 'year', 'tokens', 'ttr', 'flesch', 'polarity']
#     print(df[display_cols].round(3))
    
#     # Show semantic trends
#     print("\n🎯 Semantic Category Trends:")
#     for category in ['authority', 'identity', 'freedom', 'feminism']:
#         density_col = f'{category}_density'
#         if density_col in df.columns:
#             trend = df[density_col].corr(df['year'])
#             print(f"{category.title()}: {trend:.3f} correlation with time")
    
#     # Generate visualizations
#     print("\n📈 Generating Visualizations...")
#     plot_semantic_trends(df)
#     plot_linguistic_evolution(df)
#     plot_pronoun_trends(df)
    
#     # Show semantic clusters for latest text
#     latest_record = data[-1]
#     if 'semantic_clusters' in latest_record:
#         print("\n🔗 Semantic Clusters (2014 Text):")
#         for category, clusters in latest_record['semantic_clusters'].items():
#             if clusters:
#                 words = [word for word, count in clusters]
#                 print(f"{category}: {', '.join(words[:3])}")
    
#     print("\n✅ Analysis Complete!")
#     return df

# if __name__ == '__main__':
#     results_df = main(), '', text, flags=re.MULTILINE)
    
#     return text.strip()

# # Load actual book texts
# TEXTS = {}
# print("📚 Loading Book Files...")
# for label, filepath in BOOK_PATHS.items():
#     raw_text = load_book_text(filepath)
#     if raw_text:
#         TEXTS[label] = preprocess_book_text(raw_text)
#     else:
#         print(f"⚠️  Using fallback sample for {label}")
#         # Fallback samples if files unavailable
#         fallback_texts = {
#             "1853_Bartleby": "I am a rather elderly man. The nature of my avocations for the last thirty years has brought me into more than ordinary contact with what would seem an interesting and somewhat singular set of men, of whom as yet nothing that I know of has ever been written:—I mean the law-copyists or scriveners.",
#             "1892_Yellow": "It is very seldom that mere ordinary people like John and myself secure ancestral halls for the summer. A colonial mansion, a hereditary estate, I would say a haunted house, and reach the height of romantic felicity—but that would be asking too much of fate!",
#             "1929_Passing": "It was the last letter in Irene Redfield's little pile of morning mail. After her other correspondence, all of it brief—responses to invitations, thanks for flowers, that sort of thing—the long envelope of thin Italian paper with its almost illegible scrawl seemed out of place and alien.",
#             "1988_Small": "If you go to Antigua as a tourist, this is what you will see. If you come by aeroplane, you will land at V.C. Bird International Airport. Vere Cornwall Bird is the Prime Minister of Antigua. You may be the sort of tourist who would wonder why a Prime Minister would want an airport named after him.",
#             "2014_Fem": "We should all be feminists. My own definition of a feminist is a man or a woman who says, 'Yes, there's a problem with gender as it is today and we must fix it, we must do better.' All of us, women and men, must do better."
#         }
#         TEXTS[label] = fallback_texts.get(label, "")

# print(f"📖 Successfully loaded {len([t for t in TEXTS.values() if t])} texts for analysis")

# # Target word categories for semantic analysis
# SEMANTIC_CATEGORIES = {
#     'Authority': ['power', 'control', 'authority', 'dominance', 'rule', 'command', 'order'],
#     'Identity': ['identity', 'self', 'race', 'racial', 'gender', 'woman', 'man', 'black', 'white'],
#     'Freedom': ['freedom', 'liberty', 'independence', 'autonomy', 'choice', 'rights', 'equality'],
#     'Colonial': ['colonial', 'colonialism', 'empire', 'imperial', 'dominion', 'exploitation'],
#     'Feminism': ['feminism', 'feminist', 'gender', 'equality', 'women', 'patriarchy'],
#     'Technology': ['technology', 'digital', 'internet', 'computer', 'modern', 'innovation']
# }

# # Utility functions
# def clean_tokens(text):
#     tokens = word_tokenize(text.lower())
#     return [w for w in tokens if w.isalpha() and w not in STOP and len(w) > 2]

# def safe_ratio(num, den):
#     return num / den if den > 0 else 0

# # Core analysis functions
# def linguistic_metrics(text, tokens):
#     """Calculate key linguistic features."""
#     sents = sent_tokenize(text)
#     avg_sent_len = safe_ratio(sum(len(word_tokenize(s)) for s in sents), len(sents))
    
#     # Complexity measures
#     flesch = textstat.flesch_reading_ease(text)
#     ttr = safe_ratio(len(set(tokens)), len(tokens))  # Type-token ratio
    
#     # Syntactic features
#     tags = pos_tag(tokens)
#     adj_count = sum(1 for _, tag in tags if tag.startswith('JJ'))
#     verb_count = sum(1 for _, tag in tags if tag.startswith('VB'))
    
#     return {
#         'avg_sent_len': avg_sent_len,
#         'flesch': flesch,
#         'ttr': ttr,
#         'adj_ratio': safe_ratio(adj_count, len(tokens)),
#         'verb_ratio': safe_ratio(verb_count, len(tokens))
#     }

# def semantic_analysis(tokens):
#     """Analyze semantic categories and target words."""
#     results = {}
    
#     # Count words in each semantic category
#     for category, words in SEMANTIC_CATEGORIES.items():
#         count = sum(tokens.count(word) for word in words)
#         results[f'{category.lower()}_count'] = count
#         results[f'{category.lower()}_density'] = safe_ratio(count, len(tokens)) * 1000
    
#     # Semantic clustering - words that co-occur with target terms
#     semantic_clusters = {}
#     for category, target_words in SEMANTIC_CATEGORIES.items():
#         cluster_words = []
#         for i, token in enumerate(tokens):
#             if token in target_words:
#                 # Get context window (5 words before and after)
#                 start = max(0, i-5)
#                 end = min(len(tokens), i+6)
#                 context = tokens[start:end]
#                 cluster_words.extend([w for w in context if w not in target_words])
        
#         if cluster_words:
#             semantic_clusters[category] = Counter(cluster_words).most_common(3)
    
#     results['semantic_clusters'] = semantic_clusters
#     return results

# def sentiment_evolution(text):
#     """Analyze sentiment with TextBlob and VADER."""
#     tb_sentiment = TextBlob(text).sentiment
#     vader_scores = VADER.polarity_scores(text)
    
#     return {
#         'polarity': tb_sentiment.polarity,
#         'subjectivity': tb_sentiment.subjectivity,
#         'vader_compound': vader_scores['compound'],
#         'vader_positive': vader_scores['pos'],
#         'vader_negative': vader_scores['neg']
#     }

# def pronoun_analysis(tokens):
#     """Analyze pronoun usage patterns."""
#     pronouns = {
#         'first_person': ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'],
#         'second_person': ['you', 'your', 'yours'],
#         'third_person': ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their']
#     }
    
#     results = {}
#     for category, pron_list in pronouns.items():
#         count = sum(tokens.count(p) for p in pron_list)
#         results[f'{category}_count'] = count
#         results[f'{category}_ratio'] = safe_ratio(count, len(tokens))
    
#     return results

# # Main analysis function
# def analyze_text(text, label):
#     """Comprehensive text analysis."""
#     tokens = clean_tokens(text)
#     year = int(label.split('_')[0])
    
#     # Base metrics
#     record = {
#         'label': label,
#         'year': year,
#         'tokens': len(tokens),
#         'unique_tokens': len(set(tokens))
#     }
    
#     # Linguistic analysis
#     record.update(linguistic_metrics(text, tokens))
    
#     # Semantic analysis
#     record.update(semantic_analysis(tokens))
    
#     # Sentiment analysis
#     record.update(sentiment_evolution(text))
    
#     # Pronoun analysis
#     record.update(pronoun_analysis(tokens))
    
#     return record

# # Visualization functions
# def plot_semantic_trends(df):
#     """Plot semantic category trends over time."""
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#     axes = axes.flatten()
    
#     categories = ['authority', 'identity', 'freedom', 'colonial', 'feminism', 'technology']
    
#     for i, category in enumerate(categories):
#         density_col = f'{category}_density'
#         if density_col in df.columns:
#             axes[i].plot(df['year'], df[density_col], 'o-', linewidth=2, markersize=6)
#             axes[i].set_title(f'{category.title()} Terms', fontweight='bold')
#             axes[i].set_xlabel('Year')
#             axes[i].set_ylabel('Density (per 1000 words)')
#             axes[i].grid(True, alpha=0.3)
            
#             # Add trend line
#             z = np.polyfit(df['year'], df[density_col], 1)
#             p = np.poly1d(z)
#             axes[i].plot(df['year'], p(df['year']), '--', alpha=0.7, color='red')
    
#     plt.tight_layout()
#     plt.suptitle('Semantic Category Evolution (1850-2020)', fontsize=16, fontweight='bold', y=1.02)
#     plt.show()

# def plot_linguistic_evolution(df):
#     """Plot linguistic complexity over time."""
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
#     # Sentence complexity
#     ax1.plot(df['year'], df['avg_sent_len'], 'o-', color='blue', linewidth=2)
#     ax1.set_title('Average Sentence Length', fontweight='bold')
#     ax1.set_ylabel('Words per sentence')
#     ax1.grid(True, alpha=0.3)
    
#     # Readability
#     ax2.plot(df['year'], df['flesch'], 'o-', color='green', linewidth=2)
#     ax2.set_title('Flesch Reading Ease', fontweight='bold')
#     ax2.set_ylabel('Reading ease score')
#     ax2.grid(True, alpha=0.3)
    
#     # Vocabulary richness
#     ax3.plot(df['year'], df['ttr'], 'o-', color='orange', linewidth=2)
#     ax3.set_title('Type-Token Ratio', fontweight='bold')
#     ax3.set_ylabel('Vocabulary richness')
#     ax3.set_xlabel('Year')
#     ax3.grid(True, alpha=0.3)
    
#     # Sentiment evolution
#     ax4.plot(df['year'], df['polarity'], 'o-', color='purple', linewidth=2, label='Polarity')
#     ax4.plot(df['year'], df['vader_compound'], 's--', color='red', linewidth=2, label='VADER')
#     ax4.set_title('Sentiment Evolution', fontweight='bold')
#     ax4.set_ylabel('Sentiment score')
#     ax4.set_xlabel('Year')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# def plot_pronoun_trends(df):
#     """Plot pronoun usage trends."""
#     plt.figure(figsize=(10, 6))
    
#     pronoun_types = ['first_person_ratio', 'second_person_ratio', 'third_person_ratio']
#     colors = ['blue', 'green', 'red']
#     labels = ['First Person', 'Second Person', 'Third Person']
    
#     for ptype, color, label in zip(pronoun_types, colors, labels):
#         if ptype in df.columns:
#             plt.plot(df['year'], df[ptype], 'o-', color=color, 
#                     linewidth=2, markersize=6, label=label)
    
#     plt.title('Pronoun Usage Evolution', fontsize=14, fontweight='bold')
#     plt.xlabel('Year', fontweight='bold')
#     plt.ylabel('Ratio (pronouns/total words)', fontweight='bold')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# # Main execution
# def main():
#     print("🔍 Language Evolution Analysis: Semantic & Linguistic Trends")
#     print("=" * 60)
    
#     # Analyze all texts
#     data = []
#     for label, text in TEXTS.items():
#         print(f"📖 Processing {label}...")
#         record = analyze_text(text, label)
#         data.append(record)
    
#     # Create DataFrame and sort by year
#     df = pd.DataFrame(data).sort_values('year')
    
#     # Display key results
#     print("\n📊 Key Metrics Summary:")
#     display_cols = ['label', 'year', 'tokens', 'ttr', 'flesch', 'polarity']
#     print(df[display_cols].round(3))
    
#     # Show semantic trends
#     print("\n🎯 Semantic Category Trends:")
#     for category in ['authority', 'identity', 'freedom', 'feminism']:
#         density_col = f'{category}_density'
#         if density_col in df.columns:
#             trend = df[density_col].corr(df['year'])
#             print(f"{category.title()}: {trend:.3f} correlation with time")
    
#     # Generate visualizations
#     print("\n📈 Generating Visualizations...")
#     plot_semantic_trends(df)
#     plot_linguistic_evolution(df)
#     plot_pronoun_trends(df)
    
#     # Show semantic clusters for latest text
#     latest_record = data[-1]
#     if 'semantic_clusters' in latest_record:
#         print("\n🔗 Semantic Clusters (2014 Text):")
#         for category, clusters in latest_record['semantic_clusters'].items():
#             if clusters:
#                 words = [word for word, count in clusters]
#                 print(f"{category}: {', '.join(words[:3])}")
    
#     print("\n✅ Analysis Complete!")
#     return df

# if __name__ == '__main__':
#     results_df = main()
