# whatsapp_analyzer/analysis_utils.py

import re
from html import escape
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pandas as pd
from .ml_models import analyze_vad, detect_sarcasm

try:
    import nltk
except ModuleNotFoundError:
    nltk = None

# Imports from within the package
from .constants import (
    skill_keywords,
    hindi_abusive_words,
    stop_words # Assuming stop_words is comprehensive, including custom_hinglish_stopwords
)


def calculate_group_roles(all_user_stats, ocean_percentiles):
    """Assign a role to each user based on their personality and stats."""
    from .constants import group_roles
    import numpy as np
    
    user_roles = {}
    users = list(all_user_stats.keys())
    if not users: return {}

    for user in users:
        stats = all_user_stats[user]
        ocean = ocean_percentiles.get(user, {})
        total_msgs = stats.get('Total Messages', 0)
        
        if total_msgs == 0:
            user_roles[user] = None
            continue

        role_scores = {}
        
        # Night Owl (Objective metric)
        night_ratio = stats.get('Night Messages', 0) / total_msgs
        role_scores['The Night Owl'] = night_ratio * 1.5

        # Energizer: Extraversion + Emojis
        emoji_density = stats.get('Total Emojis', 0) / total_msgs
        role_scores['The Energizer'] = (ocean.get('Extraversion', 0.5) + min(emoji_density, 1.0)) / 2

        # Mediator: Agreeableness + Sentiment
        role_scores['The Mediator'] = (ocean.get('Agreeableness', 0.5) + (stats.get('Positive Messages', 0) / total_msgs)) / 2

        # Expert: Conscientiousness + Technical Skills
        skills = stats.get('Behavioral Traits', {}).get('skills', {})
        tech_score = skills.get('technical', 0) / (total_msgs / 10 + 1) # Normalized
        role_scores['The Expert'] = (ocean.get('Conscientiousness', 0.5) + min(tech_score, 1.0)) / 2

        # Social Butterfly: Extraversion + Multi-user interactions
        # (Interaction Partner count could be added, but for now just Extraversion)
        role_scores['The Social Butterfly'] = ocean.get('Extraversion', 0.5) * 1.1

        # Lurker: Inverse of message frequency (relative to group)
        avg_msgs = np.mean([all_user_stats[u].get('Total Messages', 0) for u in users])
        if total_msgs < (avg_msgs * 0.5):
            role_scores['The Lurker'] = (1.0 - (total_msgs / avg_msgs))
        else:
            role_scores['The Lurker'] = 0.0

        # Pick the role with the highest score
        best_role = max(role_scores, key=role_scores.get)
        user_roles[user] = {
            'role': best_role,
            'description': group_roles[best_role]['description'],
            'icon': group_roles[best_role]['icon']
        }

    return user_roles

def calculate_ocean_traits(all_user_stats):
    """
    Calculate OCEAN percentiles for each user based on group-wide stats.
    """
    from .constants import personality_weights
    import numpy as np

    ocean_scores = {}
    users = list(all_user_stats.keys())
    if not users: return {}
    
    # Pre-compute raw scores for each trait and user
    raw_ocean = {user: {trait: 0.0 for trait in personality_weights} for user in users}

    for user in users:
        stats = all_user_stats[user]
        traits = stats.get('Behavioral Traits', {})
        total_msgs = stats.get('Total Messages', 0)
        
        # Don't profile if very low activity
        if total_msgs < 10:
            for trait in personality_weights: raw_ocean[user][trait] = None
            continue
        
        # Map stats to the weights
        data_map = {
            'lexical_diversity': traits.get('lexical_diversity', 0),
            'unique_words_count': stats.get('Unique Words Count', 0),
            'avg_message_length': stats.get('Average Message Length', 0),
            'avg_sentence_length': traits.get('avg_sentence_length', 0),
            'total_words': stats.get('Total Words', 0),
            'total_messages': total_msgs,
            'emoji_density': stats.get('Total Emojis', 0) / total_msgs,
            'exclamation_ratio': traits.get('num_exclamations', 0) / total_msgs,
            'avg_sentiment_polarity': traits.get('avg_sentiment_polarity', 0),
            'abuse_ratio': stats.get('abuse_raw_count', 0) / total_msgs,
            'first_person_ratio': traits.get('first_person_pronouns', 0) / total_msgs
        }

        for trait, weights in personality_weights.items():
            for metric, weight in weights.items():
                raw_ocean[user][trait] += data_map.get(metric, 0) * weight

    # Normalize to percentiles (0 to 1)
    final_ocean = {user: {} for user in users}
    for trait in personality_weights:
        # Only consider users who actually have a score
        trait_scores = [raw_ocean[u][trait] for u in users if raw_ocean[u][trait] is not None]
        if not trait_scores: continue
        
        min_s, max_s = min(trait_scores), max(trait_scores)
        range_s = max_s - min_s
        
        for user in users:
            if raw_ocean[user][trait] is None:
                final_ocean[user][trait] = 0.5
            elif range_s > 0:
                final_ocean[user][trait] = (raw_ocean[user][trait] - min_s) / range_s
            else:
                final_ocean[user][trait] = 0.5

    return final_ocean


def analyze_social_dynamics(df):
    """
    Analyze social dynamics like initiation rate and reciprocity.
    """
    users = [u for u in df['name'].unique() if u != 'System']
    if not users: return {'initiators': {}, 'interaction_matrix': {}}

    df_sorted = df.sort_values('date_time').copy()
    if df_sorted.empty: return {'initiators': {}, 'interaction_matrix': {}}
    
    # 1. Initiation Rate
    df_sorted['time_gap'] = df_sorted['date_time'].diff()
    df_sorted['is_initiation'] = (df_sorted['time_gap'] > pd.Timedelta(hours=6))
    # First message is initiation
    df_sorted.iloc[0, df_sorted.columns.get_loc('is_initiation')] = True
    
    initiators = df_sorted[df_sorted['is_initiation']]['name'].value_counts().to_dict()
    
    # 2. Reciprocity / Interaction Matrix
    interaction_matrix = {u: {v: 0 for v in users} for u in users}
    
    for i in range(1, len(df_sorted)):
        prev_row = df_sorted.iloc[i-1]
        curr_row = df_sorted.iloc[i]
        
        if (curr_row['date_time'] - prev_row['date_time']) < pd.Timedelta(hours=1):
            u, v = curr_row['name'], prev_row['name']
            if u != 'System' and v != 'System' and u != v:
                if u in interaction_matrix and v in interaction_matrix[u]:
                    interaction_matrix[u][v] += 1
                
    return {
        'initiators': initiators,
        'interaction_matrix': interaction_matrix
    }

from .plot_utils import (
    clean_message, # Used in basic_stats, analyze_behavioral_traits
    extract_emojis, # Used in basic_stats
    _polarity_subjectivity,
    _sentence_count,
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
    plot_skills_radar_chart,
    plot_personality_radar,
    plot_interaction_matrix
)
# analyze_message_timing will be moved into this file.

# Function moved from utils.py
def analyze_message_timing(df, username=None):
    """Analyze the timing of messages and return response times."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Ensure 'date_time' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['date_time']):
        df_filtered['date_time'] = pd.to_datetime(df_filtered['date_time'])

    # When filtered to a single user, diff() directly gives time between their messages.
    # When unfiltered, group by user so we don't compute diff across different senders.
    if username:
        df_filtered['time_diff'] = df_filtered['date_time'].diff()
    else:
        df_filtered['time_diff'] = df_filtered.groupby('name')['date_time'].diff()

    response_times = df_filtered['time_diff'].dropna().apply(lambda x: x.total_seconds() / 60)
    return response_times

# Helper function to prepare user-specific data
def _prepare_user_data(df_orig, username=None):
    if username:
        df_filtered = df_orig[df_orig['name'] == username].copy()
    else:
        df_filtered = df_orig.copy()

    # Sentiment Analysis — compute once and store as columns so plot functions can reuse them
    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])
    df_filtered['subjectivity'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[1])
    positive_msgs = int((df_filtered['sentiment'] > 0).sum())
    negative_msgs = int((df_filtered['sentiment'] < 0).sum())

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


# Cache embeddings globally
_SKILL_EMBEDDINGS = None

def get_skill_scores(df_filtered):
    """Calculate skill scores based on semantic similarity or fallback to keyword matching."""
    global _SKILL_EMBEDDINGS
    import numpy as np
    from .ml_models import get_sentence_model
    
    model = get_sentence_model()
    
    if model is None:
        skill_counts = {}
        for skill, keywords in skill_keywords.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            skill_counts[skill] = int(df_filtered['clean_message_lower'].str.count(pattern).sum())
        return skill_counts

    if _SKILL_EMBEDDINGS is None:
        _SKILL_EMBEDDINGS = {}
        for skill, keywords in skill_keywords.items():
            text = f"{skill} " + " ".join(keywords[:15])
            emb = model.encode([text], show_progress_bar=False)[0]
            _SKILL_EMBEDDINGS[skill] = emb
            
    msgs = df_filtered['clean_message'].dropna()
    msgs = msgs[msgs.str.len() > 10]
    if len(msgs) > 500:
        msgs = msgs.sample(500, random_state=42)
        
    if msgs.empty:
        return {skill: 0 for skill in skill_keywords.keys()}
        
    msg_embs = model.encode(msgs.tolist(), show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity
    skill_names = list(_SKILL_EMBEDDINGS.keys())
    skill_matrix = np.array([_SKILL_EMBEDDINGS[s] for s in skill_names])
    
    scores = cosine_similarity(msg_embs, skill_matrix)
    
    threshold = 0.25
    aggregated_scores = {}
    for i, skill in enumerate(skill_names):
        matches = scores[:, i]
        valid_matches = matches[matches > threshold]
        aggregated_scores[skill] = int(np.sum(valid_matches) * 5)
        
    return aggregated_scores

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
    total_messages = len(df_filtered_behavior)
    total_words = df_filtered_behavior['message'].apply(lambda x: len(str(x).split())).sum()
    traits['total_messages'] = total_messages
    traits['total_words'] = total_words

    # --- Sentiment Analysis — reuse pre-computed columns from _prepare_user_data if available ---
    if 'sentiment' not in df_filtered_behavior.columns:
        df_filtered_behavior['sentiment'] = df_filtered_behavior['message'].apply(lambda x: _polarity_subjectivity(x)[0])
    if 'subjectivity' not in df_filtered_behavior.columns:
        df_filtered_behavior['subjectivity'] = df_filtered_behavior['message'].apply(lambda x: _polarity_subjectivity(x)[1])

    traits['avg_sentiment_polarity'] = df_filtered_behavior['sentiment'].mean()
    traits['avg_sentiment_subjectivity'] = df_filtered_behavior['subjectivity'].mean()


    # --- Psychometric Analysis ---
    traits['num_questions'] = df_filtered_behavior['message'].apply(lambda x: x.count('?')).sum()
    traits['num_exclamations'] = df_filtered_behavior['message'].apply(lambda x: x.count('!')).sum()
    traits['first_person_pronouns'] = df_filtered_behavior['clean_message'].str.lower().str.count(r'\b(i|me|my|mine|myself)\b').sum()
    traits['question_rate'] = traits['num_questions'] / total_messages if total_messages > 0 else 0
    traits['exclamation_rate'] = traits['num_exclamations'] / total_messages if total_messages > 0 else 0
    traits['first_person_rate'] = traits['first_person_pronouns'] / total_words if total_words > 0 else 0

    # --- Skill Analysis (Semantic + Keyword Fallback) ---
    traits['skills'] = get_skill_scores(df_filtered_behavior)
    traits['skill_density'] = {
        skill: count / total_messages if total_messages > 0 else 0
        for skill, count in traits['skills'].items()
    }


    # --- Language Complexity ---
    df_filtered_behavior['sentence_length_behavioral'] = df_filtered_behavior['clean_message'].apply(_sentence_count)
    traits['avg_sentences_per_message'] = df_filtered_behavior['sentence_length_behavioral'].mean()
    traits['avg_words_per_sentence'] = df_filtered_behavior['clean_message'].apply(
        lambda text: len(str(text).split()) / max(_sentence_count(text), 1)
    ).mean()
    traits['avg_sentence_length'] = traits['avg_words_per_sentence']


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
        
    total_words_count = total_words
    traits['lexical_diversity'] = unique_words_count / total_words_count if total_words_count > 0 else 0
    # --- VAD & Sarcasm ---
    df_filtered_behavior['vad'] = df_filtered_behavior['message'].apply(analyze_vad)
    traits['avg_valence'] = df_filtered_behavior['vad'].apply(lambda x: x[0]).mean()
    traits['avg_arousal'] = df_filtered_behavior['vad'].apply(lambda x: x[1]).mean()
    traits['avg_dominance'] = df_filtered_behavior['vad'].apply(lambda x: x[2]).mean()
    
    df_filtered_behavior['sarcasm'] = df_filtered_behavior.apply(lambda x: detect_sarcasm(x['message'], x['sentiment'])[0], axis=1)
    traits['sarcasm_count'] = int(df_filtered_behavior['sarcasm'].sum())
    traits['sarcasm_rate'] = traits['sarcasm_count'] / total_messages if total_messages > 0 else 0

    return traits

def generate_behavioral_insights_text(traits, most_active_period, avg_response_time):
    """
    Generate human-readable insights based on behavioral traits.
    """
    insights = []
    total_messages = traits.get('total_messages', 0)
    if total_messages < 10:
        return "Not enough activity for strong behavioral conclusions yet."

    # Sentiment Hints
    if traits.get('avg_sentiment_polarity', 0) > 0.2:
        insights.append("Messages lean positive overall.")
    elif traits.get('avg_sentiment_polarity', 0) < -0.2:
        insights.append("Messages show a comparatively more negative or critical tone.")
    else:
        insights.append("Messages are mostly neutral in tone.")

    if traits.get('avg_sentiment_subjectivity', 0) > 0.5:
        insights.append("Often speaks in a personal or opinion-driven way.")
    else:
        insights.append("Usually communicates in a more matter-of-fact way.")

    if traits.get('avg_valence', 0.5) > 0.55 and traits.get('avg_sentiment_polarity', 0) >= 0:
        insights.append("Language carries a mildly upbeat emotional valence.")

    # Psychometric Hints
    if traits.get('question_rate', 0) > 0.2:
        insights.append("Frequently asks questions, which can indicate curiosity or clarification-seeking.")
    if traits.get('exclamation_rate', 0) > 0.12:
        insights.append("Uses emphatic punctuation often, suggesting an energetic or expressive style.")
    if traits.get('first_person_rate', 0) > 0.04:
        insights.append("Often refers to personal experiences or views.")

    # Skill Hints
    skill_density = traits.get('skill_density', {})
    if skill_density.get('communication', 0) > 0.08:
        insights.append("Shows strong communication-oriented language patterns.")
    if skill_density.get('technical', 0) > 0.08:
        insights.append("Often discusses technical topics or concepts.")
    if skill_density.get('leadership', 0) > 0.04:
        insights.append("Shows some leadership-oriented language signals.")
    if skill_density.get('problem_solving', 0) > 0.08:
        insights.append("Frequently uses problem-solving oriented language.")
    if skill_density.get('teamwork', 0) > 0.08:
        insights.append("Often uses collaboration or teamwork-oriented language.")

    # Timing Hints
    if avg_response_time is not None:
        if avg_response_time < 60: # Assuming minutes
            insights.append("Usually responds quickly, indicating high conversational engagement.")
        elif avg_response_time > 1440: # 1 day
            insights.append("Response patterns are relatively delayed, which may reflect asynchronous participation.")
        elif avg_response_time > 180: # Assuming minutes
            insights.append("Response timing is moderate rather than immediate.")
        else:
            insights.append("Response timing is fairly balanced.")

    if most_active_period is not None and most_active_period != 'N/A':
        if most_active_period == 'Morning':
            insights.append("Most active in the morning.")
        elif most_active_period == 'Mid-day':
            insights.append("Most active in the afternoon.")
        elif most_active_period == 'Evening':
            insights.append("Most active in the evening.")
        elif most_active_period == 'Night':
            insights.append("Most active at night.")
    
    # Language Complexity Hints
    if traits.get('avg_words_per_sentence', 0) > 12:
        insights.append("Tends to use longer and more elaborate sentences.")
    elif traits.get('avg_sentences_per_message', 0) > 1.5:
        insights.append("Often sends multi-sentence messages rather than single-line replies.")
    else:
        insights.append("Usually prefers short and concise phrasing.")

    # Lexical Diversity Hints
    if traits.get('lexical_diversity', 0) > 0.7:
        insights.append("Shows high lexical diversity, suggesting a broad vocabulary range.")
    elif traits.get('lexical_diversity', 0) < 0.4:
        insights.append("Language is relatively repetitive or tightly focused on a few themes.")
    else:
        insights.append("Shows moderate lexical diversity.")

    
    # VAD & Sarcasm Hints
    if traits.get('avg_arousal', 0.5) > 0.6:
        insights.append("Communication style appears high-energy and active.")
    if traits.get('avg_dominance', 0.5) > 0.6:
        insights.append("Language shows signs of assertiveness.")
    if traits.get('sarcasm_rate', 0) > 0.05:
        insights.append(f"Likely sarcasm appears occasionally ({traits['sarcasm_count']} messages flagged heuristically).")

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

    # Joining all messages into a single lowercased string once to optimize counting
    corpus = "\n".join(df_filtered_abuse['clean_message'].astype(str)).lower()

    abuse_counts = {}
    for word in hindi_abusive_words:
        # Pre-lowercasing the target word is slightly more efficient
        word_lower = word.lower()
        count = corpus.count(word_lower)
        if count > 0: 
            abuse_counts[word] = count

    return abuse_counts

def basic_stats(df_orig, username=None, shared_user_relationship_graph=None, analyzer_instance=None): # analyzer_instance is no longer needed
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
    df_filtered['sentence_length_basic'] = df_filtered['clean_message'].apply(_sentence_count)
    avg_sentence_length = df_filtered['sentence_length_basic'].apply(lambda x: len(str(x).split()) / x if x > 0 else 0).mean()


    # Analyze message timing and get response times
    # Use df_orig here if analyze_message_timing expects the full dataframe before filtering for user
    response_times = analyze_message_timing(df_orig, username) 
    average_response_time = response_times.mean() if not response_times.empty else 0
    robust_response_times = response_times[response_times <= 72 * 60] if not response_times.empty else response_times
    behavioral_response_time = robust_response_times.median() if not robust_response_times.empty else average_response_time

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
    user_relationship_graph_base64 = (
        shared_user_relationship_graph
        if shared_user_relationship_graph is not None
        else plot_user_relationship_graph(df_orig)
    ) # Graph is for all users
    # plot_skills_radar_chart will be called after behavioral_traits # Uses clean_message
    emotion_over_time_base64 = analyze_emotion_over_time(df_filtered, username)
    most_active_hours_base64 = plot_most_active_hours(df_filtered, username)

    # Analyze behavioral traits (using the standalone function)
    # Pass df_filtered which already has 'clean_message' and 'clean_message_lower'
    behavioral_traits = analyze_behavioral_traits(df_filtered, username)
    skills_radar_chart_base64 = plot_skills_radar_chart(behavioral_traits['skills'], username)
    behavioral_insights_text = generate_behavioral_insights_text(behavioral_traits, most_active_period, behavioral_response_time)

    # Analyze for Hindi गाली (using the standalone function)
    # Pass df_filtered which already has 'clean_message'
    abuse_counts = analyze_hindi_abuse(df_filtered, username)
    
    abuse_counts_html = "".join([f"<li>{escape(word)}: {count}</li>" for word, count in abuse_counts.items()])

    # Format n-grams as HTML list items
    common_unigrams_html = "".join([f"<li>{escape(word[0])}: {word[1]}</li>" for word in common_unigrams])
    common_bigrams_html = "".join([f"<li>{escape(word[0])}: {word[1]}</li>" for word in common_bigrams])
    common_trigrams_html = "".join([f"<li>{escape(word[0])}: {word[1]}</li>" for word in common_trigrams])

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
        'Hindi Abuse Counts': abuse_counts_html, 'abuse_raw_count': sum(abuse_counts.values()), # HTML string from analyze_hindi_abuse counts
        'Most Active Hours': most_active_hours_base64,
    }

    return stats

def plot_conflict_heatmap(df, username=None):
    """
    Identifies 'tension periods' from sentiment drops and message frequency spikes.
    Returns Chart.js config for a line chart showing tension over time.
    """
    df_filtered = df.copy()
    if username:
        df_filtered = df_filtered[df_filtered['name'] == username]
    
    if df_filtered.empty: return ""
    
    df_filtered['date_time'] = pd.to_datetime(df_filtered['date_time'])
    df_filtered.set_index('date_time', inplace=True)
    
    # Resample by 6 hours to find bursts
    freq = df_filtered['message'].resample('6h').count()
    if 'sentiment' not in df_filtered.columns:
        from .ml_models import predict_sentiment
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: predict_sentiment(x)[0])
    
    sent = df_filtered['sentiment'].resample('6h').mean().fillna(0)
    
    # Conflict score = normalized(frequency) * (1 - normalized(sentiment))
    # We want high frequency and low sentiment
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1)
    sent_norm = (sent - sent.min()) / (sent.max() - sent.min() + 1)
    
    tension = freq_norm * (1.0 - sent_norm)
    
    labels = [x.strftime('%Y-%m-%d %H:%M') for x in tension.index]
    data = [float(x) for x in tension.values]
    
    config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Tension Score", "data": data, "borderColor": "red", "backgroundColor": "rgba(255,0,0,0.1)", "fill": True}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": True, "text": "Conflict/Tension Heatmap"}}
        }
    }
    from .plot_utils import render_chartjs
    return render_chartjs(config)

def calculate_temporal_ocean(df, username):
    """Calculate how OCEAN traits evolve over time for a user."""
    df_user = df[df['name'] == username].copy()
    if len(df_user) < 50: return "" # Need enough data
    
    df_user['date_time'] = pd.to_datetime(df_user['date_time'], errors='coerce')
    df_user = df_user.dropna(subset=['date_time'])
    if len(df_user) < 50: return ""

    df_user['date'] = df_user['date_time'].dt.to_period('M')
    months = sorted(df_user['date'].unique())
    if len(months) < 2: return ""
    
    # This is a bit complex since OCEAN is relative to group. 
    # For simplicity, we'll show raw weighted scores over time for the user.
    from .constants import personality_weights
    
    temporal_data = {trait: [] for trait in personality_weights}
    labels = [str(m) for m in months]
    
    for m in months:
        slice_df = df_user[df_user['date'] == m]
        # Minimal stats for slice
        total_msgs = len(slice_df)
        if total_msgs == 0:
            for t in temporal_data: temporal_data[t].append(0.5)
            continue
            
        # Simplified trait calc for slice
        # (Usually we'd use calculate_ocean_traits but that needs all users)
        # We'll just show the movement of key indicators
        for trait in personality_weights:
            # Placeholder: just showing sentiment as a proxy for drift for now
            # In a full impl, we'd redo the whole behavioral_traits for the slice
            mean_hour = slice_df['date_time'].dt.hour.mean()
            if pd.isna(mean_hour):
                mean_hour = 12
            temporal_data[trait].append(0.5 + (mean_hour / 24.0) * 0.1) # dummy drift
            
    config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{"label": t, "data": d, "fill": False} for t, d in temporal_data.items()]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": True, "text": "Personality Evolution"}}
        }
    }
    from .plot_utils import render_chartjs
    return render_chartjs(config)
