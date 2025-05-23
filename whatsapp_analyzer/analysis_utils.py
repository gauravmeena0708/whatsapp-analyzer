# whatsapp_analyzer/analysis_utils.py

import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd # Though df is passed, good practice

# Imports from within the package
from .constants import (
    skill_keywords,
    hindi_abusive_words,
    stop_words # Assuming stop_words is comprehensive, including custom_hinglish_stopwords
)
from .plot_utils import (
    clean_message, # Used in basic_stats, analyze_behavioral_traits
    extract_emojis, # Used in basic_stats
    plot_activity_heatmap,
    plot_sentiment_distribution,
    plot_most_active_hours,
    generate_wordcloud,
    analyze_language_complexity,
    plot_response_time_distribution,
    analyze_sentiment_over_time,
    analyze_emotion_over_time,
    plot_emoji_usage,
    plot_sentiment_bubble,
    plot_vocabulary_diversity,
    plot_language_complexity_pos,
    plot_user_relationship_graph,
    plot_skills_radar_chart
)
# analyze_message_timing will be moved into this file.

# Function moved from utils.py
def analyze_message_timing(df, username=None):
    """Analyze the timing of messages and return response times."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
        df_filtered['date'] = pd.to_datetime(df_filtered['date'])

    # Use 'date_time' column for accurate time difference calculation
    df_filtered['time_diff'] = df_filtered.groupby('name')['date_time'].diff()
    response_times = df_filtered['time_diff'].dropna().apply(lambda x: x.total_seconds() / 60)  # in minutes

    return response_times

# Helper function to prepare user-specific data
def _prepare_user_data(df_orig, username=None):
    if username:
        df_filtered = df_orig[df_orig['name'] == username].copy()
    else:
        df_filtered = df_orig.copy()

    # Sentiment Analysis
    sentiments = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    positive_msgs = sum(sentiments > 0)
    negative_msgs = sum(sentiments < 0)

    # Time of Day Analysis
    def categorize_time_of_day(hour):
        if 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 16: return 'Mid-day'
        elif 16 <= hour < 18: return 'Evening'
        else: return 'Night'

    df_filtered['time_of_day'] = df_filtered['hour'].apply(categorize_time_of_day)
    morning_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Morning'])
    midday_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Mid-day'])
    evening_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Evening'])
    night_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Night'])
    message_counts_by_period = {'Morning': morning_msgs, 'Mid-day': midday_msgs, 'Evening': evening_msgs, 'Night': night_msgs}
    most_active_period = max(message_counts_by_period, key=message_counts_by_period.get) if message_counts_by_period else "N/A"

    # Clean messages - this is crucial for many subsequent analyses
    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower()
    
    return df_filtered, positive_msgs, negative_msgs, morning_msgs, midday_msgs, evening_msgs, night_msgs, most_active_period

# Helper function to calculate n-grams
def _calculate_ngrams(df_filtered):
    # Most Common n-grams
    def get_top_ngrams(corpus, n=1, top_k=10):
        # Ensure corpus is not empty and contains actual text
        if corpus.empty or corpus.str.strip().empty:
            return []
        try:
            vec = CountVectorizer(ngram_range=(n, n), stop_words=list(stop_words)).fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            # Filter for counts > 1
            words_freq_filtered = [item for item in words_freq if item[1] > 1]
            words_freq = sorted(words_freq_filtered, key=lambda x: x[1], reverse=True)
            return words_freq[:top_k]
        except ValueError: # Handles cases where vocabulary might be empty after stop word removal
            return []

    # Ensure there's data to process for n-grams
    corpus_for_ngrams = df_filtered['clean_message_lower'].dropna()
    common_unigrams = get_top_ngrams(corpus_for_ngrams, 1, 10)
    common_bigrams = get_top_ngrams(corpus_for_ngrams, 2, 10)
    common_trigrams = get_top_ngrams(corpus_for_ngrams, 3, 10)
    
    return common_unigrams, common_bigrams, common_trigrams

def analyze_behavioral_traits(df, username=None):
    """
    Analyze behavioral traits and return a dictionary of insights.
    Assumes df already has 'clean_message' and 'clean_message_lower' columns.
    """
    if username:
        # df_filtered is a copy, so modifications are safe.
        df_filtered_behavior = df[df['name'] == username].copy() 
    else:
        df_filtered_behavior = df.copy()

    # Ensure 'clean_message' column exists if not already present from _prepare_user_data
    if 'clean_message' not in df_filtered_behavior.columns:
        df_filtered_behavior['clean_message'] = df_filtered_behavior['message'].apply(lambda x: clean_message(str(x)))
    if 'clean_message_lower' not in df_filtered_behavior.columns:
        df_filtered_behavior['clean_message_lower'] = df_filtered_behavior['clean_message'].str.lower()

    traits = {}

    # --- Sentiment Analysis (already done in _prepare_user_data, but if called independently) ---
    if 'sentiment_polarity' not in df_filtered_behavior.columns:
        df_filtered_behavior['sentiment_polarity'] = df_filtered_behavior['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    if 'sentiment_subjectivity' not in df_filtered_behavior.columns:
        df_filtered_behavior['sentiment_subjectivity'] = df_filtered_behavior['message'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    
    traits['avg_sentiment_polarity'] = df_filtered_behavior['sentiment_polarity'].mean()
    traits['avg_sentiment_subjectivity'] = df_filtered_behavior['sentiment_subjectivity'].mean()


    # --- Psychometric Analysis ---
    traits['num_questions'] = df_filtered_behavior['message'].apply(lambda x: x.count('?')).sum()
    traits['num_exclamations'] = df_filtered_behavior['message'].apply(lambda x: x.count('!')).sum()
    traits['first_person_pronouns'] = df_filtered_behavior['clean_message'].str.lower().str.count(r'\b(i|me|my|mine|myself)\b').sum()

    # --- Skill Analysis (Keyword-based) ---
    traits['skills'] = {}
    for skill, keywords in skill_keywords.items():
        traits['skills'][skill] = sum(df_filtered_behavior['clean_message_lower'].str.count('|'.join(keywords)))


    # --- Language Complexity ---
    df_filtered_behavior['sentence_length_behavioral'] = df_filtered_behavior['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))) if str(x).strip() else 0)
    traits['avg_sentence_length'] = df_filtered_behavior['sentence_length_behavioral'].mean()


    # --- Lexical Diversity ---
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    corpus_for_lexical = df_filtered_behavior['clean_message_lower'].dropna()
    unique_words_count = 0
    if not corpus_for_lexical.empty:
        try:
            word_matrix = vectorizer.fit_transform(corpus_for_lexical)
            unique_words_count = len(vectorizer.get_feature_names_out())
        except ValueError: 
            unique_words_count = 0 
        
    total_words_count = df_filtered_behavior['message'].apply(lambda x: len(str(x).split())).sum()
    traits['lexical_diversity'] = unique_words_count / total_words_count if total_words_count > 0 else 0

    return traits

def generate_behavioral_insights_text(traits, most_active_period, avg_response_time):
    """
    Generate human-readable insights based on behavioral traits.
    """
    insights = []

    # Sentiment Hints
    if traits.get('avg_sentiment_polarity', 0) > 0.2:
        insights.append("Tends to express positive sentiment in messages.")
    elif traits.get('avg_sentiment_polarity', 0) < -0.2:
        insights.append("Tends to express negative sentiment in messages.")
    else:
        insights.append("Maintains a neutral tone in messages.")

    if traits.get('avg_sentiment_subjectivity', 0) > 0.5:
        insights.append("Expresses subjective opinions and evaluations.")
    else:
        insights.append("Tends to communicate more objectively.")

    # Psychometric Hints
    if traits.get('num_questions', 0) > 20:
        insights.append("Asks a lot of questions, possibly indicating curiosity or a need for clarification.")
    if traits.get('num_exclamations', 0) > 5:
        insights.append("Uses exclamations frequently, suggesting excitement or strong opinions.")
    if traits.get('first_person_pronouns', 0) > 10:
        insights.append("Often refers to themselves, which might indicate a focus on personal experiences or opinions.")

    # Skill Hints
    user_skills = traits.get('skills', {})
    if user_skills.get('communication', 0) > 5:
        insights.append("Demonstrates strong communication skills based on keyword analysis.")
    if user_skills.get('technical', 0) > 5:
        insights.append("Exhibits technical skills based on keyword analysis.")
    if user_skills.get('leadership', 0) > 2:
        insights.append("Shows potential leadership qualities based on keyword analysis.")
    if user_skills.get('problem_solving', 0) > 5:
        insights.append("Appears to have good problem-solving skills based on keyword analysis.")
    if user_skills.get('teamwork', 0) > 5:
        insights.append("Likely a good team player based on keyword analysis.")

    # Timing Hints
    if avg_response_time is not None:
        if avg_response_time < 60: # Assuming minutes
            insights.append("Responds quickly to messages, indicating high engagement.")
        elif avg_response_time > 180: # Assuming minutes
            insights.append("Takes longer to respond, suggesting lower engagement or a busy schedule.")
        else:
            insights.append("Has a moderate response time.")

    if most_active_period is not None:
        if most_active_period == 'Morning':
            insights.append("Most active in the morning.")
        elif most_active_period == 'Mid-day':
            insights.append("Most active in the afternoon.")
        elif most_active_period == 'Evening':
            insights.append("Most active in the evening.")
        else: # Night
            insights.append("Most active at night.")
    
    # Language Complexity Hints
    if traits.get('avg_sentence_length', 0) > 3: # Assuming this is words per sentence from analyze_behavioral_traits
        insights.append("Uses long and complex sentences.")
    else:
        insights.append("Uses short and concise sentences.")

    # Lexical Diversity Hints
    if traits.get('lexical_diversity', 0) > 0.7:
        insights.append("Exhibits high lexical diversity, indicating a broad vocabulary.")
    elif traits.get('lexical_diversity', 0) < 0.4:
        insights.append("Has low lexical diversity, suggesting a more repetitive or focused communication style.")
    else:
        insights.append("Shows moderate lexical diversity.")

    return "<br/>".join(insights)

def analyze_hindi_abuse(df, username=None):
    """
    Analyze the use of Hindi abusive words and return a dictionary of counts.
    Assumes df has 'clean_message'.
    """
    if username:
        df_filtered_abuse = df[df['name'] == username].copy()
    else:
        df_filtered_abuse = df.copy()
    
    # Ensure 'clean_message' column exists
    if 'clean_message' not in df_filtered_abuse.columns:
        df_filtered_abuse['clean_message'] = df_filtered_abuse['message'].apply(lambda x: clean_message(str(x)))

    abuse_counts = {}
    for word in hindi_abusive_words:
        count = df_filtered_abuse['clean_message'].str.lower().str.count(word).sum()
        if count > 0: 
            abuse_counts[word] = count

    return abuse_counts

def basic_stats(df_orig, username=None, analyzer_instance=None): # analyzer_instance is no longer needed
    """
    Calculate basic statistics about messages, including sentiment, time analysis,
    most common n-grams (unigrams, bigrams, trigrams), most active period, and visualizations.
    """
    # Prepare user-specific data and initial stats
    df_filtered, positive_msgs, negative_msgs, morning_msgs, \
    midday_msgs, evening_msgs, night_msgs, most_active_period = \
        _prepare_user_data(df_orig, username)

    # Calculate N-grams
    common_unigrams, common_bigrams, common_trigrams = _calculate_ngrams(df_filtered)
    
    # Unique words count (using df_filtered which has clean_message_lower)
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    unique_words_count = 0
    corpus_for_unique_words = df_filtered['clean_message_lower'].dropna()
    if not corpus_for_unique_words.empty:
        try:
            word_matrix = vectorizer.fit_transform(corpus_for_unique_words)
            unique_words_count = len(vectorizer.get_feature_names_out())
        except ValueError: 
            unique_words_count = 0

    # Top 5 Emojis (using df_filtered)
    df_filtered['emojis_list'] = df_filtered['message'].apply(extract_emojis) # Renamed to avoid conflict with df_filtered['emoji'] from basic_cleanup
    all_emojis_list = [emoji_item for sublist in df_filtered['emojis_list'] for emoji_item in sublist]
    top_5_emojis = Counter(all_emojis_list).most_common(5)

    # Average Sentence Length
    df_filtered['sentence_length_basic'] = df_filtered['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))))
    avg_sentence_length = df_filtered['sentence_length_basic'].apply(lambda x: len(str(x).split()) / x if x > 0 else 0).mean()


    # Analyze message timing and get response times
    # Use df_orig here if analyze_message_timing expects the full dataframe before filtering for user
    response_times = analyze_message_timing(df_orig, username) 
    average_response_time = response_times.mean() if not response_times.empty else 0

    # Visualizations (ensure df_filtered is passed, not df_orig for user-specific plots)
    activity_heatmap_base64 = plot_activity_heatmap(df_filtered, username)
    sentiment_distribution_base64 = plot_sentiment_distribution(df_filtered, username)
    wordcloud_base64 = generate_wordcloud(df_filtered, username) # Uses clean_message
    language_complexity_base64 = analyze_language_complexity(df_filtered, username) # Uses clean_message
    response_time_distribution_base64 = plot_response_time_distribution(response_times, username)
    sentiment_over_time_base64 = analyze_sentiment_over_time(df_filtered, username) # Use df_filtered
    emoji_usage_base64 = plot_emoji_usage(df_filtered, username) # Uses 'emojis_list' now
    sentiment_bubble_base64 = plot_sentiment_bubble(df_filtered, username)
    vocabulary_diversity_base64 = plot_vocabulary_diversity(df_filtered, username) # Uses clean_message_lower
    language_complexity_pos_base64 = plot_language_complexity_pos(df_filtered, username)
    user_relationship_graph_base64 = plot_user_relationship_graph(df_orig) # Graph is for all users
    skills_radar_chart_base64 = plot_skills_radar_chart(df_filtered, username) # Uses clean_message
    emotion_over_time_base64 = analyze_emotion_over_time(df_filtered, username)
    most_active_hours_base64 = plot_most_active_hours(df_filtered, username)

    # Analyze behavioral traits (using the standalone function)
    # Pass df_filtered which already has 'clean_message' and 'clean_message_lower'
    behavioral_traits = analyze_behavioral_traits(df_filtered, username)
    behavioral_insights_text = generate_behavioral_insights_text(behavioral_traits, most_active_period, average_response_time)

    # Analyze for Hindi गाली (using the standalone function)
    # Pass df_filtered which already has 'clean_message'
    abuse_counts = analyze_hindi_abuse(df_filtered, username)
    
    abuse_counts_html = "<ul>"
    for word, count in abuse_counts.items():
        abuse_counts_html += f"<li>{word}: {count}</li>"
    abuse_counts_html += "</ul>"

    # Format n-grams as HTML list items
    common_unigrams_html = "".join([f"<li>{word[0]}: {word[1]}</li>" for word in common_unigrams])
    common_bigrams_html = "".join([f"<li>{word[0]}: {word[1]}</li>" for word in common_bigrams])
    common_trigrams_html = "".join([f"<li>{word[0]}: {word[1]}</li>" for word in common_trigrams])

    stats = {
        'Total Messages': len(df_filtered),
        'Total Words': df_filtered['message'].apply(lambda x: len(str(x).split())).sum(),
        'Unique Users': df_filtered['name'].nunique(), # Should be 1 if username is specified
        'Total Emojis': df_filtered['emojicount'].sum() if 'emojicount' in df_filtered.columns else 0,
        'Total URLs': df_filtered['urlcount'].sum() if 'urlcount' in df_filtered.columns else 0,
        'Total YouTube URLs': df_filtered['yturlcount'].sum() if 'yturlcount' in df_filtered.columns else 0,
        'Total Media': df_filtered['mediacount'].sum() if 'mediacount' in df_filtered.columns else 0,
        'Total Edits': df_filtered['editcount'].sum() if 'editcount' in df_filtered.columns else 0,
        'Total Deletes': df_filtered['deletecount'].sum() if 'deletecount' in df_filtered.columns else 0,
        'Average Message Length': df_filtered['mlen'].mean() if 'mlen' in df_filtered.columns else 0,
        'Positive Messages': positive_msgs,
        'Negative Messages': negative_msgs,
        'Morning Messages': morning_msgs,
        'Mid-day Messages': midday_msgs,
        'Evening Messages': evening_msgs,
        'Night Messages': night_msgs,
        'Most Active Period': most_active_period,
        'Unique Words Count': unique_words_count,
        'Common Unigrams': common_unigrams_html, # Store HTML string
        'Common Bigrams': common_bigrams_html,   # Store HTML string
        'Common Trigrams': common_trigrams_html, # Store HTML string
        'Top 5 Emojis': top_5_emojis, # Keep as list of tuples for now, will be processed in analyzer.py
        'Average Sentence Length': avg_sentence_length,
        'Average Response Time': average_response_time,
        'Activity Heatmap': activity_heatmap_base64,
        'Sentiment Distribution': sentiment_distribution_base64,
        'Word Cloud': wordcloud_base64,
        'Language Complexity': language_complexity_base64,
        'Response Time Distribution': response_time_distribution_base64,
        'Sentiment Over Time': sentiment_over_time_base64,
        'Emoji Usage': emoji_usage_base64,
        'Sentiment Bubble': sentiment_bubble_base64,
        'Vocabulary Diversity': vocabulary_diversity_base64,
        'Language Complexity POS': language_complexity_pos_base64,
        'User Relationship Graph': user_relationship_graph_base64,
        'Skills Radar Chart': skills_radar_chart_base64,
        'Behavioral Traits': behavioral_traits, # Dictionary from analyze_behavioral_traits
        'Emotion Over Time': emotion_over_time_base64,
        'Behavioral Insights Text': behavioral_insights_text, # Text from generate_behavioral_insights_text
        'Hindi Abuse Counts': abuse_counts_html, # HTML string from analyze_hindi_abuse counts
        'Most Active Hours': most_active_hours_base64,
    }

    return stats
