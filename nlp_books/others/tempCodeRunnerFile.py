# # Comprehensive Analysis: Evolution of Language & Culture (1850–2020)
# # Tools: Python, NLTK, Pandas, Matplotlib, Seaborn, TextBlob, VADER, NRCLex, textstat, spaCy

# import os, re
# import nltk
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from textblob import TextBlob
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
# from nltk import pos_tag, Text
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nrclex import NRCLex
# import textstat
# import spacy
# from collections import Counter

# # Initialize resources
# nltk.download('punkt'); nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger'); nltk.download('vader_lexicon')
# STOP = set(stopwords.words('english'))
# VADER = SentimentIntensityAnalyzer()
# NLP = spacy.load('en_core_web_sm')

# # Configuration
# TEXTS = {
#     "1853_Bartleby": "books/bartleby_1853.txt",
#     "1892_Yellow":   "books/yellow_1892.txt",
#     "1929_Passing":  "books/nella_1929.txt",
#     "1988_Small":    "books/small_place_1988.txt",
#     "2014_Fem":      "books/feminist_2014.txt"
# }
# UK_US_PAIRS = [('honour', 'honor'), ('labour', 'labor'), ('defence', 'defense')]
# KEY_TERMS = ['freedom', 'race', 'feminism', 'colonial', 'technology']
# TOPIC_LEXICONS = {
#     'Globalization': ['global', 'trade', 'capitalism'],
#     'Identity': ['identity', 'race', 'gender'],
#     'Environment': ['climate', 'earth', 'pollution']
# }

# # Utility functions

# def load_text(path):
#     with open(path, encoding='utf-8') as f:
#         return f.read()

# def clean_tokens(text):
#     return [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in STOP]

# # Morphological analysis

# def morphological_diversity(text):
#     doc = NLP(text)
#     lemmas = [token.lemma_ for token in doc if token.is_alpha]
#     return len(set(lemmas)) / len(lemmas) if lemmas else 0

# def morph_tag_distribution(text):
#     doc = NLP(text)
#     morphs = [token.morph.to_str() for token in doc if token.is_alpha]
#     return Counter(morphs)

# # Core analyses

# def sentence_stats(text):
#     sents = sent_tokenize(text)
#     lengths = [len(word_tokenize(s)) for s in sents]
#     return sum(lengths) / len(lengths) if lengths else 0

# def passive_rate(text):
#     sents = sent_tokenize(text); count=0
#     for s in sents:
#         tags = pos_tag(word_tokenize(s))
#         if any(w.lower() in ('is','are','was','were','be','been','being') and nxt=='VBN'
#                for (w,_),( _,nxt) in zip(tags,tags[1:])):
#             count+=1
#     return count/len(sents) if sents else 0

# def negation_rate(tokens):
#     negs=[w for w in tokens if re.fullmatch(r"not|never|no|none|n't",w)]
#     return len(negs)/len(tokens) if tokens else 0

# def dialogue_ratio(text):
#     quotes=re.findall(r'"(.*?)"',text)
#     qwords=sum(len(q.split()) for q in quotes)
#     total=len(word_tokenize(text))
#     return qwords/total if total else 0

# def pov_ratio(tokens):
#     first=tokens.count('i')+tokens.count('we')
#     third=tokens.count('he')+tokens.count('she')+tokens.count('they')
#     return first/third if third else 0

# def pos_diversity(tokens, prefix):
#     tags=pos_tag(tokens)
#     words=[w for w,t in tags if t.startswith(prefix)]
#     return len(set(words))/len(words) if words else 0

# def sentiment_scores(text):
#     return TextBlob(text).sentiment.polarity, VADER.polarity_scores(text)['compound']

# def emotion_profile(tokens):
#     counts=Counter()
#     for w in tokens:
#         counts.update(NRCLex(w).raw_emotion_scores)
#     return counts

# def named_entities(text):
#     doc=NLP(text)
#     return Counter(ent.label_ for ent in doc.ents)

# def readability(text):
#     return textstat.flesch_reading_ease(text), textstat.gunning_fog(text)

# def topic_trends(tokens):
#     return {t: sum(tokens.count(w) for w in ws) for t,ws in TOPIC_LEXICONS.items()}

# # Analyze each text

# def analyze(path, label, data):
#     text=load_text(path)
#     tokens=clean_tokens(text)
#     record={'Label':label,'Tokens':len(tokens),'Unique':len(set(tokens))}

#     # Linguistic metrics
#     record['AvgSentLen']=sentence_stats(text)
#     record['PassiveRate']=passive_rate(text)
#     record['NegRate']=negation_rate(tokens)
#     record['DialogueRatio']=dialogue_ratio(text)
#     record['POVRatio']=pov_ratio(tokens)
#     record['AdjDiv']=pos_diversity(tokens,'JJ')
#     record['VerbDiv']=pos_diversity(tokens,'VB')
#     record['MorphDiv']=morphological_diversity(text)
#     # Sentiment
#     tb,vd=sentiment_scores(text)
#     record['TBPolarity'],record['VADER']=tb,vd
#     # Readability
#     fr,fg=readability(text)
#     record['Flesch'],record['Fog']=fr,fg
#     # Pronouns
#     for p in ['i','we','you','he','she','they']:
#         record[p]=tokens.count(p)
#     # Emotion
#     emo=emotion_profile(tokens)
#     for e in ['joy','anger','fear','sadness']:
#         record[e]=emo.get(e,0)
#     # Entities
#     ne=named_entities(text)
#     for ent,c in ne.items(): record[f'ENT_{ent}']=c
#     # Spelling variants
#     for uk,us in UK_US_PAIRS:
#         record[uk]=tokens.count(uk)
#         record[us]=tokens.count(us)
#     # Key terms & topics
#     for t in KEY_TERMS: record[t]=tokens.count(t)
#     record.update(topic_trends(tokens))

#     # Morphological tag distribution for later plotting
#     data.append(record)

# # Plotting

# def plot_series(df,x,y,title,annotate=False):
#     plt.figure(figsize=(10,5))
#     sns.lineplot(data=df,x=x,y=y,marker='o')
#     plt.title(title)
#     plt.xticks(rotation=45)
#     if annotate:
#         for xi,yi in zip(df[x],df[y]): plt.text(xi, yi*1.05, f'{yi:.2f}', ha='center')
#     plt.tight_layout();plt.show()

# def plot_bar(df,x,y,title):
#     plt.figure(figsize=(10,5))
#     sns.barplot(data=df,x=x,y=y)
#     plt.title(title)
#     plt.xticks(rotation=45)
#     plt.tight_layout();plt.show()

# # Execute
# if __name__=='__main__':
#     data=[]
#     for lbl,path in TEXTS.items(): analyze(path,lbl,data)
#     df=pd.DataFrame(data)
#     print(df)

#     # Numeric plots
#     metrics_line=['AvgSentLen','PassiveRate','NegRate','DialogueRatio','POVRatio','TBPolarity','VADER','Flesch','Fog']
#     for m in metrics_line: plot_series(df,'Label',m,m.replace('Rate',' Rate').replace('Avg','Average ').replace('TB','TextBlob ').replace('VADER','VADER ').replace('Flesch','Flesch ').replace('Fog','Gunning Fog ').title(),True)
#     # POS & Morph
#     plot_bar(df,'Label','AdjDiv','Adjective Diversity')
#     plot_bar(df,'Label','VerbDiv','Verb Diversity')
#     plot_bar(df,'Label','MorphDiv','Morphological Diversity')
#     # Pronouns
#     pron_df=df.melt(id_vars='Label',value_vars=['i','we','you','he','she','they'],var_name='Pronoun',value_name='Count')
#     plot_bar(pron_df,'Label','Count','Pronoun Usage')
#     # Spelling
#     for uk,us in UK_US_PAIRS:
#         pair_df=df.melt(id_vars='Label',value_vars=[uk,us],var_name='Variant',value_name='Count')
#         plot_bar(pair_df,'Label','Count',f'Spelling {uk} vs {us}')
#     # Topics
#     for t in TOPIC_LEXICONS: plot_series(df,'Label',t,f'Theme: {t}',True)
#     # Emotions
#     for e in ['joy','anger','fear','sadness']: plot_series(df,'Label',e,f'Emotion: {e}',True)
#     # Morph tags sample from largest text
#     largest= max(TEXTS.keys(), key=lambda k: df.loc[df['Label']==k,'Tokens'].values[0])
#     dist=morph_tag_distribution(load_text(TEXTS[largest]))
#     morph_df=pd.DataFrame(dist.most_common(10),columns=['Morph','Count'])
#     plot_bar(morph_df,'Morph','Count','Top Morphological Tags')
#     # Named Entities summary
#     ent_cols=[c for c in df.columns if c.startswith('ENT_')]
#     print('\nNamed Entities:')
#     print(df[['Label']+ent_cols])
#     # Concordance
#     print("\nConcordance Examples (2014_Fem):")
#     txt=Text(word_tokenize(load_text(TEXTS['2014_Fem']).lower()))
#     for t in KEY_TERMS:
#         print(f"-- {t} --"); txt.concordance(t,width=60,lines=2)

# Enhanced Comprehensive Analysis: Evolution of Language & Culture (1850–2020)
# Tools: Python, NLTK, Pandas, Matplotlib, Seaborn, TextBlob, VADER, NRCLex, textstat, spaCy

import os, re, warnings
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag, Text
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    NRCLEX_AVAILABLE = False
    print("Warning: NRCLex not available. Emotion analysis will be skipped.")

import textstat
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Some advanced features will be skipped.")

from collections import Counter
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize resources with error handling
def setup_nltk():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

setup_nltk()

# Global variables
STOP = set(stopwords.words('english'))
VADER = SentimentIntensityAnalyzer()

# Enhanced styling for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Sample texts (since we don't have actual files)
SAMPLE_TEXTS = {
    "1853_Bartleby": """I am a rather elderly man. The nature of my avocations for the last thirty years has brought me into more than ordinary contact with what would seem an interesting and somewhat singular set of men, of whom as yet nothing that I know of has ever been written:—I mean the law-copyists or scriveners. I have known very many of them, professionally and privately, and if I pleased, could relate divers histories, at which good-natured gentlemen might smile, and sentimental souls might weep. But I waive the biographies of all other scriveners for a few passages in the life of Bartleby, who was a scrivener the strangest I ever saw or heard of.""",
    
    "1892_Yellow": """It is very seldom that mere ordinary people like John and myself secure ancestral halls for the summer. A colonial mansion, a hereditary estate, I would say a haunted house, and reach the height of romantic felicity—but that would be asking too much of fate! Still I will proudly declare that there is something queer about it. Else, why should it be let so cheaply? And why have stood so long untenanted? John laughs at me, so of course I take pains to control myself before him, at least, and that makes me very tired.""",
    
    "1929_Passing": """It was the last letter in Irene Redfield's little pile of morning mail. After her other correspondence, all of it brief—responses to invitations, thanks for flowers, that sort of thing—the long envelope of thin Italian paper with its almost illegible scrawl seemed out of place and alien. And there was, too, something mysterious and slightly furtive about it. A thin sly thing which bore no return address to betray the sender. Not that she hadn't immediately known who its sender was. Some two years ago she had one very like it in outward appearance. Furtive, but yet in some peculiar, determined way a little flaunting.""",
    
    "1988_Small": """If you go to Antigua as a tourist, this is what you will see. If you come by aeroplane, you will land at V.C. Bird International Airport. Vere Cornwall Bird is the Prime Minister of Antigua. You may be the sort of tourist who would wonder why a Prime Minister would want an airport named after him—why not a school, why not a hospital, why not some great public monument? You are a tourist and you have not yet seen the hospital in Antigua, you have not yet seen a school in Antigua, you have not yet seen the public monuments in Antigua.""",
    
    "2014_Fem": """We should all be feminists. My own definition of a feminist is a man or a woman who says, 'Yes, there's a problem with gender as it is today and we must fix it, we must do better.' All of us, women and men, must do better. Gender as it exists today is a grave injustice. I am angry. We should all be angry. Anger has a long history of bringing about positive change. But I am also hopeful, because I believe deeply in the ability of human beings to remake themselves for the better."""
}

# Configuration
UK_US_PAIRS = [('honour', 'honor'), ('labour', 'labor'), ('defence', 'defense')]
KEY_TERMS = ['freedom', 'race', 'feminism', 'colonial', 'technology']
TOPIC_LEXICONS = {
    'Globalization': ['global', 'trade', 'capitalism', 'international', 'world'],
    'Identity': ['identity', 'race', 'gender', 'self', 'personal'],
    'Environment': ['climate', 'earth', 'pollution', 'nature', 'environmental']
}

# Utility functions
def clean_tokens(text):
    """Clean and tokenize text, removing stopwords and non-alphabetic tokens."""
    tokens = word_tokenize(text.lower())
    return [w for w in tokens if w.isalpha() and w not in STOP and len(w) > 2]

def safe_divide(numerator, denominator):
    """Safe division to avoid division by zero."""
    return numerator / denominator if denominator > 0 else 0

# Enhanced morphological analysis
def morphological_diversity(text):
    """Calculate morphological diversity using spaCy if available."""
    if not SPACY_AVAILABLE:
        return 0
    doc = nlp(text[:1000000])  # Limit text length for performance
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    return safe_divide(len(set(lemmas)), len(lemmas))

# Core linguistic analyses
def sentence_stats(text):
    """Calculate average sentence length."""
    try:
        sents = sent_tokenize(text)
        if not sents:
            return 0
        lengths = [len(word_tokenize(s)) for s in sents]
        return safe_divide(sum(lengths), len(lengths))
    except:
        return 0

def passive_rate(text):
    """Calculate passive voice rate."""
    try:
        sents = sent_tokenize(text)[:100]  # Limit for performance
        if not sents:
            return 0
        
        count = 0
        for s in sents:
            tokens = word_tokenize(s.lower())
            tags = pos_tag(tokens)
            
            # Look for auxiliary + past participle pattern
            for i in range(len(tags) - 1):
                word, tag = tags[i]
                next_word, next_tag = tags[i + 1]
                if (word in ('is', 'are', 'was', 'were', 'be', 'been', 'being') and 
                    next_tag == 'VBN'):
                    count += 1
                    break
        
        return safe_divide(count, len(sents))
    except:
        return 0

def negation_rate(tokens):
    """Calculate negation rate."""
    neg_patterns = ['not', 'never', 'no', 'none', 'nothing', 'nobody', 'nowhere']
    neg_contractions = [w for w in tokens if w.endswith("n't")]
    neg_words = [w for w in tokens if w in neg_patterns]
    return safe_divide(len(neg_words) + len(neg_contractions), len(tokens))

def dialogue_ratio(text):
    """Calculate dialogue to total text ratio."""
    try:
        quotes = re.findall(r'"([^"]*)"', text)
        quote_words = sum(len(q.split()) for q in quotes)
        total_words = len(word_tokenize(text))
        return safe_divide(quote_words, total_words)
    except:
        return 0

def pov_ratio(tokens):
    """Calculate first person to third person pronoun ratio."""
    first_person = tokens.count('i') + tokens.count('we') + tokens.count('me') + tokens.count('us')
    third_person = tokens.count('he') + tokens.count('she') + tokens.count('they') + tokens.count('him') + tokens.count('her') + tokens.count('them')
    return safe_divide(first_person, third_person) if third_person > 0 else first_person

def pos_diversity(tokens, prefix):
    """Calculate part-of-speech diversity."""
    try:
        tags = pos_tag(tokens)
        words = [w for w, t in tags if t.startswith(prefix)]
        return safe_divide(len(set(words)), len(words))
    except:
        return 0

def sentiment_scores(text):
    """Calculate sentiment using TextBlob and VADER."""
    try:
        tb_polarity = TextBlob(text).sentiment.polarity
        vader_compound = VADER.polarity_scores(text)['compound']
        return tb_polarity, vader_compound
    except:
        return 0, 0

def emotion_profile(tokens):
    """Calculate emotion profile using NRCLex if available."""
    if not NRCLEX_AVAILABLE:
        return Counter()
    
    try:
        emotion_counts = Counter()
        # Sample tokens for performance
        sample_tokens = tokens[:500] if len(tokens) > 500 else tokens
        
        for word in sample_tokens:
            try:
                emotion_scores = NRCLex(word).raw_emotion_scores
                emotion_counts.update(emotion_scores)
            except:
                continue
        return emotion_counts
    except:
        return Counter()

def named_entities(text):
    """Extract named entities using spaCy if available."""
    if not SPACY_AVAILABLE:
        return Counter()
    
    try:
        # Limit text length for performance
        doc = nlp(text[:100000])
        return Counter(ent.label_ for ent in doc.ents)
    except:
        return Counter()

def readability_scores(text):
    """Calculate readability scores."""
    try:
        flesch = textstat.flesch_reading_ease(text)
        fog = textstat.gunning_fog(text)
        return flesch, fog
    except:
        return 0, 0

def topic_trends(tokens):
    """Calculate topic-specific word counts."""
    return {topic: sum(tokens.count(word) for word in words) 
            for topic, words in TOPIC_LEXICONS.items()}

# Main analysis function
def analyze_text(text, label):
    """Comprehensive text analysis."""
    tokens = clean_tokens(text)
    
    record = {
        'Label': label,
        'Year': int(label.split('_')[0]),
        'Tokens': len(tokens),
        'Unique': len(set(tokens)),
        'TTR': safe_divide(len(set(tokens)), len(tokens))  # Type-Token Ratio
    }

    # Linguistic metrics
    record['AvgSentLen'] = sentence_stats(text)
    record['PassiveRate'] = passive_rate(text)
    record['NegRate'] = negation_rate(tokens)
    record['DialogueRatio'] = dialogue_ratio(text)
    record['POVRatio'] = pov_ratio(tokens)
    record['AdjDiv'] = pos_diversity(tokens, 'JJ')
    record['VerbDiv'] = pos_diversity(tokens, 'VB')
    record['MorphDiv'] = morphological_diversity(text)

    # Sentiment analysis
    tb_polarity, vader_compound = sentiment_scores(text)
    record['TBPolarity'] = tb_polarity
    record['VADER'] = vader_compound

    # Readability
    flesch, fog = readability_scores(text)
    record['Flesch'] = flesch
    record['Fog'] = fog

    # Pronouns
    pronouns = ['i', 'we', 'you', 'he', 'she', 'they']
    for pronoun in pronouns:
        record[pronoun] = tokens.count(pronoun)

    # Emotions
    emotions = emotion_profile(tokens)
    for emotion in ['joy', 'anger', 'fear', 'sadness', 'trust', 'surprise']:
        record[emotion] = emotions.get(emotion, 0)

    # Named entities
    entities = named_entities(text)
    for entity_type in ['PERSON', 'ORG', 'GPE', 'DATE']:
        record[f'ENT_{entity_type}'] = entities.get(entity_type, 0)

    # Spelling variants
    for uk, us in UK_US_PAIRS:
        record[uk] = tokens.count(uk)
        record[us] = tokens.count(us)

    # Key terms
    for term in KEY_TERMS:
        record[term] = tokens.count(term)

    # Topic trends
    record.update(topic_trends(tokens))

    return record

# Enhanced plotting functions
def create_time_series_plot(df, metric, title):
    """Create enhanced time series plot."""
    plt.figure(figsize=(12, 6))
    
    # Create the main plot
    ax = sns.lineplot(data=df, x='Year', y=metric, marker='o', linewidth=3, markersize=8)
    
    # Enhance the plot
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel(metric.replace('Rate', ' Rate').replace('Avg', 'Average '), fontsize=12, fontweight='bold')
    
    # Add value annotations
    for i, row in df.iterrows():
        plt.annotate(f'{row[metric]:.3f}', 
                    (row['Year'], row[metric]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontweight='bold')
    
    # Style improvements
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add trend line
    z = np.polyfit(df['Year'], df[metric], 1)
    p = np.poly1d(z)
    plt.plot(df['Year'], p(df['Year']), "--", alpha=0.7, color='red', linewidth=2)
    
    plt.show()

def create_comparison_plot(df, metrics, title):
    """Create comparison plot for multiple metrics."""
    plt.figure(figsize=(14, 8))
    
    # Normalize metrics for comparison
    df_norm = df.copy()
    for metric in metrics:
        if df[metric].std() != 0:
            df_norm[metric] = (df[metric] - df[metric].mean()) / df[metric].std()
    
    for metric in metrics:
        plt.plot(df['Year'], df_norm[metric], marker='o', linewidth=2, 
                label=metric.replace('Rate', ' Rate').replace('Avg', 'Average '), markersize=6)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Values', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_heatmap(df, title):
    """Create correlation heatmap."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Year']]
    
    if len(numeric_cols) < 2:
        return
    
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.2f')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def create_stacked_bar_chart(df, categories, title):
    """Create stacked bar chart for categorical data."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    bottom = np.zeros(len(df))
    colors = sns.color_palette("husl", len(categories))
    
    for i, category in enumerate(categories):
        if category in df.columns:
            plt.bar(df['Year'], df[category], bottom=bottom, 
                   label=category.title(), color=colors[i], alpha=0.8)
            bottom += df[category]
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# Execute analysis
def main():
    """Main execution function."""
    print("🔍 Analyzing Language Evolution (1850-2020)")
    print("=" * 50)
    
    # Analyze all texts
    data = []
    for label, text in SAMPLE_TEXTS.items():
        print(f"Processing {label}...")
        record = analyze_text(text, label)
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values('Year')
    
    print("\n📊 Analysis Results:")
    print(df[['Label', 'Tokens', 'Unique', 'TTR', 'AvgSentLen', 'TBPolarity']].round(3))
    
    # Create visualizations
    print("\n📈 Generating Visualizations...")
    
    # Time series plots for key metrics
    metrics_to_plot = ['AvgSentLen', 'PassiveRate', 'NegRate', 'TBPolarity', 'VADER', 'TTR']
    for metric in metrics_to_plot:
        if metric in df.columns and df[metric].notna().any():
            create_time_series_plot(df, metric, f'Evolution of {metric.replace("Rate", " Rate").replace("Avg", "Average ")} Over Time')
    
    # Comparison plots
    linguistic_metrics = ['AvgSentLen', 'PassiveRate', 'NegRate', 'TTR']
    available_metrics = [m for m in linguistic_metrics if m in df.columns]
    if len(available_metrics) >= 2:
        create_comparison_plot(df, available_metrics, 'Linguistic Metrics Comparison Over Time')
    
    sentiment_metrics = ['TBPolarity', 'VADER']
    available_sentiment = [m for m in sentiment_metrics if m in df.columns]
    if len(available_sentiment) >= 2:
        create_comparison_plot(df, available_sentiment, 'Sentiment Analysis Over Time')
    
    # Pronoun usage
    pronouns = ['i', 'we', 'you', 'he', 'she', 'they']
    available_pronouns = [p for p in pronouns if p in df.columns]
    if available_pronouns:
        create_stacked_bar_chart(df, available_pronouns, 'Pronoun Usage Evolution')
    
    # Topic trends
    topics = list(TOPIC_LEXICONS.keys())
    available_topics = [t for t in topics if t in df.columns]
    if available_topics:
        create_stacked_bar_chart(df, available_topics, 'Topic Trends Over Time')
    
    # Correlation heatmap
    create_heatmap(df, 'Feature Correlation Matrix')
    
    # Summary statistics
    print("\n📋 Summary Statistics:")
    print("-" * 30)
    print(f"Average sentence length trend: {df['AvgSentLen'].corr(df['Year']):.3f}")
    print(f"Sentiment trend: {df['TBPolarity'].corr(df['Year']):.3f}")
    print(f"Vocabulary richness trend: {df['TTR'].corr(df['Year']):.3f}")
    
    print("\n✅ Analysis Complete!")
    
    return df

if __name__ == '__main__':
    df_results = main()