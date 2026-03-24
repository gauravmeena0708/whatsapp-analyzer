# whatsapp_analyzer/plot_utils.py
import base64
import re
from io import BytesIO
from collections import Counter
from functools import lru_cache

try:
    import emoji
    EMOJI_DATA = emoji.EMOJI_DATA
except ModuleNotFoundError:
    EMOJI_DATA = {}

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
try:
    import nltk
except ModuleNotFoundError:
    nltk = None

try:
    from textblob import TextBlob
except ModuleNotFoundError:
    TextBlob = None

try:
    from wordcloud import WordCloud
except ModuleNotFoundError:
    WordCloud = None

from .constants import stop_words, skill_keywords  # stop_words already built in constants.py


def _filter_by_user(df, username):
    """Return a copy of df filtered to username, or a copy of the full df if username is None."""
    if username:
        return df[df['name'] == username].copy()
    return df.copy()

@lru_cache(maxsize=None)  # Cache all unique calls
def clean_message(msg):
    """
    Clean the message by removing URLs, media omitted phrases, and trimming spaces.
    """
    # Remove URLs
    msg = re.sub(r'http[s]?://\S+', '', msg)
    # Remove "media omitted" phrases, case-insensitive
    msg = re.sub(r'<media omitted>|\bmedia omitted\b', '', msg, flags=re.IGNORECASE)
    # Strip any extra spaces
    msg = msg.strip()
    return msg

def extract_emojis(text):
    """Extract emojis from text."""
    return [c for c in text if c in EMOJI_DATA or ord(c) > 10000]


def _sentence_count(text):
    text = str(text).strip()
    if not text:
        return 0
    if nltk is not None:
        try:
            return len(nltk.sent_tokenize(text))
        except LookupError:
            pass
    parts = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return max(len(parts), 1)


def _polarity_subjectivity(text):
    text = str(text)
    if TextBlob is not None:
        sentiment = TextBlob(text).sentiment
        return float(sentiment.polarity), float(sentiment.subjectivity)

    positive_words = {"good", "great", "happy", "love", "excellent", "awesome", "nice", "best"}
    negative_words = {"bad", "sad", "hate", "angry", "terrible", "awful", "worst", "no"}
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0, 0.0
    pos = sum(word in positive_words for word in words)
    neg = sum(word in negative_words for word in words)
    polarity = (pos - neg) / max(len(words), 1)
    subjectivity = min((pos + neg) / max(len(words), 1) * 2, 1.0)
    return polarity, subjectivity

def plot_to_base64(plt):
    """Convert a Matplotlib plot to a base64 encoded image."""
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64


def _render_empty_plot(message, title, xlabel="", ylabel=""):
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    apply_consistent_plot_styling(plt, title, xlabel, ylabel)
    return plot_to_base64(plt)

def apply_consistent_plot_styling(plt, title, xlabel, ylabel):
    """Applies consistent styling to Matplotlib plots."""
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

def plot_activity_heatmap(df, username=None):
    """Plot an activity heatmap and return base64 image."""
    df_filtered = _filter_by_user(df, username)

    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered['weekday'] = df_filtered['date'].dt.day_name()
    df_filtered['hour'] = df_filtered['hour']

    heatmap_data = df_filtered.pivot_table(index='weekday', columns='hour', values='message', aggfunc='count', fill_value=0)
    heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    plt.figure(figsize=(12, 6), constrained_layout=True)
    sns.heatmap(heatmap_data, cmap='viridis', annot=False, cbar_kws={'label': 'Number of Messages'})
    apply_consistent_plot_styling(plt, f'Activity Heatmap {"for " + username if username else ""}', 'Hour of the Day', 'Day of the Week')
    return plot_to_base64(plt)

def plot_sentiment_distribution(df, username=None):
    """Plot sentiment distribution and return base64 image."""
    df_filtered = _filter_by_user(df, username)

    if 'sentiment' not in df_filtered.columns:
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])

    plt.figure(figsize=(8, 5))
    sns.histplot(df_filtered['sentiment'], bins=20, kde=True, color='skyblue')
    apply_consistent_plot_styling(plt, f'Sentiment Distribution {"for " + username if username else ""}', 'Sentiment Polarity', 'Frequency')
    return plot_to_base64(plt)

def plot_most_active_hours(df, username=None):
    """Plot a bar chart of the most active hours and return base64 image."""
    df_filtered = _filter_by_user(df, username)

    message_counts_by_hour = df_filtered['hour'].value_counts().sort_index()

    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.bar(message_counts_by_hour.index, message_counts_by_hour.values, color='skyblue')
    apply_consistent_plot_styling(plt, f'Most Active Hours {"for " + username if username else ""}', 'Hour of the Day', 'Number of Messages')
    return plot_to_base64(plt)

def generate_wordcloud(df, username=None):
    """Generate word cloud and return base64 image."""
    df_filtered = _filter_by_user(df, username)

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    text = " ".join(msg for msg in df_filtered['clean_message'] if isinstance(msg, str) and len(msg.strip())>0)

    plt.figure(figsize=(10, 8))
    if not text: # Handle case with no text for word cloud
        plt.text(0.5, 0.5, "No words to display in word cloud.", ha='center', va='center', fontsize=12)
    else:
        try:
            if WordCloud is None:
                raise ValueError("wordcloud package is not installed")
            wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
        except ValueError as e: # Catch any other potential errors from WordCloud
            plt.text(0.5, 0.5, f"Could not generate word cloud:\n{e}", ha='center', va='center', fontsize=12, color='red')
            
    plt.axis("off")
    plt.title(f'Word Cloud {"for " + username if username else ""}', fontsize=14)
    return plot_to_base64(plt)

def analyze_language_complexity(df, username=None):
    """Analyze language complexity and return base64 images."""
    df_filtered = _filter_by_user(df, username)

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    
    # filter out emojis
    df_filtered['word_length'] = df_filtered['clean_message'].apply(
        lambda x: [len(word) for word in str(x).split() if word.lower() not in stop_words and len(word) > 1 and not all(c in EMOJI_DATA or ord(c) > 10000 for c in word)]
    )
    
    avg_word_lengths = df_filtered['word_length'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

    # Handle cases with only emojis or empty messages
    df_filtered['sentence_length'] = df_filtered['clean_message'].apply(_sentence_count)
    avg_sentence_lengths = df_filtered['sentence_length'].apply(
        lambda x: len(str(x).split()) / x if x > 0 and len(str(x).split()) > 0 else 0
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(avg_word_lengths, bins=20, kde=True, color='skyblue', ax=axs[0])
    axs[0].set_title(f'Average Word Length {"for " + username if username else ""}', fontsize=14)
    axs[0].set_xlabel('Average Word Length', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)

    sns.histplot(avg_sentence_lengths, bins=20, kde=True, color='salmon', ax=axs[1])
    axs[1].set_title(f'Average Sentence Length {"for " + username if username else ""}', fontsize=14)
    axs[1].set_xlabel('Average Sentence Length (words)', fontsize=12)
    axs[1].set_ylabel('Frequency', fontsize=12)

    # Convert the combined plot to base64
    combined_plot_base64 = plot_to_base64(plt)
    
    return combined_plot_base64

def plot_response_time_distribution(response_times, username=None):
    """Plot the distribution of response times."""
    if response_times is None or len(response_times) == 0:
        return _render_empty_plot(
            "Not enough messages to compute response times.",
            f'Response Time Distribution {"for " + username if username else ""}',
            "Response Time (minutes)",
            "Frequency",
        )

    plt.figure(figsize=(8, 5))
    sns.histplot(response_times, bins=20, kde=True, color='skyblue')
    apply_consistent_plot_styling(plt, f'Response Time Distribution {"for " + username if username else ""}', 'Response Time (minutes)', 'Frequency')
    return plot_to_base64(plt)

def analyze_sentiment_over_time(df, username=None):
    """Analyze sentiment over time and return base64 image of the plot."""
    df_filtered = _filter_by_user(df, username)
    if df_filtered.empty:
        return _render_empty_plot(
            "No messages available for sentiment trend analysis.",
            f'Sentiment Over Time {"for " + username if username else ""}',
            "Date",
            "Average Sentiment",
        )

    if 'sentiment' not in df_filtered.columns:
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    # Resample to daily frequency and calculate the mean sentiment
    daily_sentiment = df_filtered['sentiment'].resample('W').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment.index, daily_sentiment.values, color='purple')
    apply_consistent_plot_styling(plt, f'Sentiment Over Time {"for " + username if username else ""}', 'Date', 'Average Sentiment')
    
    return plot_to_base64(plt)

def analyze_emotion_over_time(df, username=None):
    """Analyze emotion over time using TextBlob and return base64 image of the plot."""
    df_filtered = _filter_by_user(df, username)
    if df_filtered.empty:
        return _render_empty_plot(
            "No messages available for emotion trend analysis.",
            f'Emotion Trends Over Time {"for " + username if username else ""}',
            "Date",
            "Emotion Score",
        )

    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    # Reuse pre-computed sentiment column if available
    if 'sentiment' not in df_filtered.columns:
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])

    # Map polarity to broad emotion buckets
    def categorize_emotion(score):
        if score > 0.5:
            return "joy"
        elif score > 0:
            return "surprise"
        elif score < -0.5:
            return "sadness"
        elif score < 0:
            return "anger"
        else:
            return "neutral"

    df_filtered['emotion'] = df_filtered['sentiment'].apply(categorize_emotion)
    
    # Resample to daily frequency and count the occurrences of each emotion
    daily_emotions = df_filtered.groupby(pd.Grouper(freq='D'))['emotion'].apply(lambda x: x.value_counts()).unstack(fill_value=0)
    
    
    plt.figure(figsize=(12, 6))
    for emotion in daily_emotions.columns:
        plt.plot(daily_emotions.index, daily_emotions[emotion], label=emotion)
    plt.legend()
    apply_consistent_plot_styling(plt, f'Emotion Trends Over Time {"for " + username if username else ""}', 'Date', 'Emotion Score')
    
    return plot_to_base64(plt)

def plot_emoji_usage(df, username=None):
    """Plot a bar chart of the top 5 emojis used and return base64 image."""
    df_filtered = _filter_by_user(df, username)

    df_filtered['emojis'] = df_filtered['message'].apply(extract_emojis)
    all_emojis = [emoji for sublist in df_filtered['emojis'] for emoji in sublist]
    top_emojis = Counter(all_emojis).most_common(5)

    if not top_emojis: # Handle case with no emojis
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No emojis found.", ha='center', va='center', fontsize=12)
        apply_consistent_plot_styling(plt, f'Emoji Usage {"for " + username if username else ""}', 'Emoji', 'Count')
    else:
        emojis, counts = zip(*top_emojis)
        plt.figure(figsize=(10, 6))
        plt.bar(emojis, counts, color='skyblue')
        apply_consistent_plot_styling(plt, f'Emoji Usage {"for " + username if username else ""}', 'Emoji', 'Count')
    return plot_to_base64(plt)

def plot_sentiment_bubble(df, username=None):
    """
    Plot a bubble chart of sentiment distribution and return base64 image.
    x-axis: Polarity (Positive/Negative)
    y-axis: Subjectivity (Objective/Subjective)
    Bubble size: Number of messages
    """
    df_filtered = _filter_by_user(df, username)

    # Reuse pre-computed sentiment columns when available
    if 'sentiment' in df_filtered.columns:
        df_filtered['polarity'] = df_filtered['sentiment']
    else:
        df_filtered['polarity'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])

    if 'subjectivity' not in df_filtered.columns:
        df_filtered['subjectivity'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[1])

    # Count the number of messages for each sentiment
    sentiment_counts = df_filtered.groupby(['polarity', 'subjectivity']).size().reset_index(name='counts')

    plt.figure(figsize=(10, 8))
    plt.scatter(sentiment_counts['polarity'], sentiment_counts['subjectivity'], s=sentiment_counts['counts']*10, alpha=0.6, color='purple')
    apply_consistent_plot_styling(plt, f'Sentiment Distribution {"for " + username if username else ""}', 'Polarity (Positive/Negative)', 'Subjectivity (Objective/Subjective)')
    return plot_to_base64(plt)

def plot_vocabulary_diversity(df, username=None):
    """
    Plot vocabulary diversity as a bar chart comparing unique vs total words,
    with the diversity ratio annotated.
    """
    df_filtered = _filter_by_user(df, username)

    if 'clean_message_lower' not in df_filtered.columns:
        if 'clean_message' not in df_filtered.columns:
            df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
        df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower()

    corpus = df_filtered['clean_message_lower'].dropna()

    unique_words_count = 0
    if not corpus.empty:
        vectorizer = CountVectorizer(stop_words=list(stop_words))
        try:
            vectorizer.fit_transform(corpus)
            unique_words_count = len(vectorizer.get_feature_names_out())
        except ValueError:
            unique_words_count = 0

    total_words_count = int(df_filtered['message'].apply(lambda x: len(str(x).split())).sum())
    diversity_ratio = round(unique_words_count / total_words_count, 3) if total_words_count > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['Unique Words', 'Total Words'], [unique_words_count, total_words_count],
                  color=['#075e54', '#25d366'])
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_title(f'Vocabulary Diversity {"for " + username if username else ""}\n(Diversity Ratio: {diversity_ratio:.3f})',
                 fontsize=13)
    ax.set_ylabel('Word Count', fontsize=12)
    ax.set_ylim(0, max(total_words_count * 1.15, 1))
    plt.tight_layout()
    return plot_to_base64(plt)

def plot_language_complexity_pos(df, username=None):
    """
    Analyze and plot the distribution of POS tags for a user or the entire chat,
    and return a base64 image of the plot.
    """
    df_filtered = _filter_by_user(df, username)

    # Function to extract POS tags from a message
    def extract_pos_tags(message):
        if TextBlob is None:
            return []
        analysis = TextBlob(message)
        return [tag for (word, tag) in analysis.tags]

    # Apply POS tag extraction to each message
    df_filtered['pos_tags'] = df_filtered['message'].apply(extract_pos_tags)

    # Flatten the list of POS tags and count their occurrences
    all_pos_tags = [tag for sublist in df_filtered['pos_tags'] for tag in sublist]
    if not all_pos_tags:
        return _render_empty_plot(
            "No POS tags available for this dataset.",
            f'POS Tag Distribution {"for " + username if username else ""}',
            "POS Tag",
            "Count",
        )

    pos_counts = Counter(all_pos_tags)

    # Convert to DataFrame for plotting
    pos_df = pd.DataFrame(list(pos_counts.items()), columns=['POS Tag', 'Count'])

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='POS Tag', y='Count', data=pos_df, color='skyblue')
    apply_consistent_plot_styling(plt, f'POS Tag Distribution {"for " + username if username else ""}', 'POS Tag', 'Count')

    return plot_to_base64(plt)

def plot_user_relationship_graph(df):
    """
    Plot a graph representing the relationships between users based on message interactions.
    Nodes represent users, and edges represent interactions between them.
    """
    df = df[df['name'] != "System"].reset_index(drop=True)
    
    # Create a graph
    G = nx.Graph()

    # Add nodes for each user
    for user in df['name'].unique():
        G.add_node(user)

    if len(G.nodes) == 0:
        return _render_empty_plot(
            "No user interactions available.",
            "User Relationship Graph",
        )

    # Vectorised interaction counting using pandas shift (avoids slow Python loop)
    names = df['name'].values
    senders = names[:-1]
    receivers = names[1:]
    # Only count cross-person consecutive messages
    mask = senders != receivers
    pair_counts = pd.Series(list(zip(senders[mask], receivers[mask]))).value_counts()
    for (sender, receiver), weight in pair_counts.items():
        if G.has_edge(sender, receiver):
            G[sender][receiver]['weight'] += weight
        else:
            G.add_edge(sender, receiver, weight=int(weight))

    if len(G.edges) == 0:
        return _render_empty_plot(
            "Not enough cross-user interactions to draw a graph.",
            "User Relationship Graph",
        )

    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=8, width=[d['weight'] / 5 for u, v, d in G.edges(data=True)])
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("User Relationship Graph", fontsize=15)

    return plot_to_base64(plt)

def plot_skills_radar_chart(df, username=None):
    """
    Generate a radar chart to visualize various skills based on keyword analysis.
    """
    df_filtered = _filter_by_user(df, username)

    

    # Count keyword occurrences for each skill (escape keywords to avoid regex issues)
    skill_counts = {}
    for skill, keywords in skill_keywords.items():
        pattern = '|'.join(re.escape(kw) for kw in keywords)
        skill_counts[skill] = int(df_filtered['clean_message'].str.lower().str.count(pattern).sum())

    # Prepare data for radar chart
    skills = list(skill_counts.keys())
    counts = list(skill_counts.values())
    
    # Normalize counts for radar chart
    if counts:
        max_val = max(counts)
        if max_val == 0: # All skill counts are zero
             # Prevent division by zero; keep normalized_counts as zeros or handle as appropriate
            normalized_counts = [0.0] * len(counts)
        else:
            normalized_counts = [c / max_val for c in counts]
    else: # No skills defined or counts list is empty for some reason
        normalized_counts = []
        skills = [] # Ensure skills is also empty if counts is empty

    # Number of variables (skills)
    num_vars = len(skills)
    if num_vars == 0:
        return _render_empty_plot(
            "No skill categories available.",
            f'Skills Radar Chart {"for " + username if username else ""}',
        )

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Initialize radar chart
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], skills, color='black', size=10)

    # Draw ylabels (normalized counts)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, normalized_counts + normalized_counts[:1], linewidth=2, linestyle='solid')
    ax.fill(angles, normalized_counts + normalized_counts[:1], 'b', alpha=0.1)

    # Add title
    plt.title(f'Skills Radar Chart {"for " + username if username else ""}', size=15, color='black', y=1.1)

    return plot_to_base64(plt)
