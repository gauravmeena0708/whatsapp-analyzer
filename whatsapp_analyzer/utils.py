# whatsapp_analyzer/utils.py
import regex
import emoji
import pandas as pd
import numpy as np
from .constants import skill_keywords, hindi_abusive_words, stop_words

URL_PATTERN = r"(https?://\S+)"
YOUTUBE_PATTERN = r"(https?://youtu(\.be|be\.com)\S+)"

def get_emojis(text):
    emoji_list = []
    data = regex.findall(r"\X", text)
    for word in data:
        if any(char in emoji.EMOJI_DATA for char in word):
            emoji_list.append(word)
    return emoji_list

def get_urls(text):
    url_list = regex.findall(URL_PATTERN, text)
    return url_list

def get_yturls(text):
    url_list = regex.findall(YOUTUBE_PATTERN, text)
    return url_list

def df_basic_cleanup(df):
    """
    Performs basic data cleaning and feature engineering on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing parsed chat data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with added features.
    """
    df["date_time"] = pd.to_datetime(df["t"])
    df["date"] = df["date_time"].dt.date
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month.astype(str).str.zfill(2)
    df["day"] = df["date_time"].dt.day

    df["dayn"] = df["date_time"].dt.day_name().astype("category")
    df["monthn"] = df["date_time"].dt.month_name()

    df["doy"] = df["date_time"].dt.dayofyear
    df["dow"] = df["date_time"].dt.dayofweek
    df["woy"] = df["date_time"].dt.isocalendar().week
    df["time"] = df["date_time"].dt.time
    df["hour"] = df["date_time"].dt.hour
    df["min"] = df["date_time"].dt.minute
    df["hm"] = df["hour"] + round(df["min"] / 60, 2)

    df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    df["yw"] = df["year"].astype(str) + "-" + df["woy"].astype(str)
    df["yd"] = df["year"].astype(str) + "-" + df["doy"].astype(str)
    df["md"] = df["monthn"].astype(str) + "-" + df["date"].astype(str)

    df["mlen"] = df["message"].str.len()

    df["emoji"] = df["message"].apply(get_emojis)
    df["emojicount"] = df["emoji"].str.len()

    df["urls"] = df["message"].apply(get_urls)
    df["urlcount"] = df["urls"].str.len()

    df["yturls"] = df["message"].apply(get_yturls)
    df["yturlcount"] = df["yturls"].str.len()

    df["mediacount"] = np.where(df["message"] == "<Media omitted>", 1, 0)
    df["editcount"] = np.where(
        df["message"].str.contains("<This message was edited>"), 1, 0
    )
    df["deletecount"] = np.where(
        (
            (df["message"] == "This message was deleted")
            | (df["message"] == "You deleted this message")
        ),
        1,
        0,
    )

    df.drop("t", inplace=True, axis=1)
    df = df[
        [
            "date_time",
            "date",
            "year",
            "month",
            "monthn",
            "day",
            "dayn",
            "woy",
            "doy",
            "dow",
            "ym",
            "yw",
            "yd",
            "md",
            "time",
            "hour",
            "min",
            "hm",
            "name",
            "message",
            "mlen",
            "emoji",
            "emojicount",
            "urls",
            "urlcount",
            "yturls",
            "yturlcount",
            "mediacount",
            "editcount",
            "deletecount",
        ]
    ]
    return df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from .constants import custom_hinglish_stopwords  # Import from constants within the package
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import os
import emoji
import base64
from io import BytesIO
from functools import lru_cache
import networkx as nx
import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt # Already imported
import warnings



# Combine NLTK stopwords with custom Hinglish stopwords
stop_words = set(stopwords.words('english')).union(custom_hinglish_stopwords)

@lru_cache(maxsize=None)  # Cache all unique calls
def clean_message(msg):
    """
    Clean the message by removing URLs, media omitted phrases, and trimming spaces.
    """
    # Remove URLs
    msg = re.sub(r'http[s]?://\S+', '', msg)
    # Remove "media omitted" phrases, case-insensitive
    msg = re.sub(r'\b(media omitted|<media omitted>)\b', '', msg, flags=re.IGNORECASE)
    # Strip any extra spaces
    msg = msg.strip()
    return msg

def extract_emojis(text):
    """Extract emojis from text."""
    return [c for c in text if c in emoji.EMOJI_DATA]

def plot_to_base64(plt):
    """Convert a Matplotlib plot to a base64 encoded image."""
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def apply_consistent_plot_styling(plt, title, xlabel, ylabel):
    """Applies consistent styling to Matplotlib plots."""
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.tight_layout()

def plot_activity_heatmap(df, username=None):
    """Plot an activity heatmap and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

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
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    plt.figure(figsize=(8, 5))
    sns.histplot(df_filtered['sentiment'], bins=20, kde=True, color='skyblue')
    apply_consistent_plot_styling(plt, f'Sentiment Distribution {"for " + username if username else ""}', 'Sentiment Polarity', 'Frequency')
    return plot_to_base64(plt)

def plot_most_active_hours(df, username=None):
    """Plot a bar chart of the most active hours and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    message_counts_by_hour = df_filtered['hour'].value_counts().sort_index()

    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.bar(message_counts_by_hour.index, message_counts_by_hour.values, color='skyblue')
    apply_consistent_plot_styling(plt, f'Most Active Hours {"for " + username if username else ""}', 'Hour of the Day', 'Number of Messages')
    return plot_to_base64(plt)

def generate_wordcloud(df, username=None):
    """Generate word cloud and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    text = " ".join(msg for msg in df_filtered['clean_message'] if isinstance(msg, str) and len(msg)>1)

    wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'Word Cloud {"for " + username if username else ""}', fontsize=14)
    #plt.tight_layout()
    return plot_to_base64(plt)

def analyze_language_complexity(df, username=None):
    """Analyze language complexity and return base64 images."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    
    # filter out emojis
    df_filtered['word_length'] = df_filtered['clean_message'].apply(
        lambda x: [len(word) for word in str(x).split() if word.lower() not in stop_words and len(word) > 1 and not all(c in emoji.EMOJI_DATA for c in word)]
    )
    
    avg_word_lengths = df_filtered['word_length'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

    # Handle cases with only emojis or empty messages
    df_filtered['sentence_length'] = df_filtered['clean_message'].apply(
        lambda x: len(nltk.sent_tokenize(str(x))) if str(x).strip() else 0
    )
    avg_sentence_lengths = df_filtered['sentence_length'].apply(
        lambda x: len(str(x).split()) / x if x > 0 and len(str(x).split()) > 0 else 0
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(avg_word_lengths, bins=20, kde=True, color='skyblue', ax=axs[0])
    apply_consistent_plot_styling(plt, f'Average Word Length {"for " + username if username else ""}', 'Average Word Length', 'Frequency')

    sns.histplot(avg_sentence_lengths, bins=20, kde=True, color='salmon', ax=axs[1])
    apply_consistent_plot_styling(plt, f'Average Sentence Length {"for " + username if username else ""}', 'Average Sentence Length (words)', 'Frequency')

    #plt.tight_layout()
    
    # Convert the combined plot to base64
    combined_plot_base64 = plot_to_base64(plt)
    
    return combined_plot_base64

def analyze_message_timing(df, username=None):
    """Analyze the timing of messages and return response times."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['time_diff'] = df_filtered.groupby('name')['date'].diff()
    response_times = df_filtered['time_diff'].dropna().apply(lambda x: x.total_seconds() / 60)  # in minutes

    return response_times

def plot_response_time_distribution(response_times, username=None):
    """Plot the distribution of response times."""
    plt.figure(figsize=(8, 5))
    sns.histplot(response_times, bins=20, kde=True, color='skyblue')
    apply_consistent_plot_styling(plt, f'Response Time Distribution {"for " + username if username else ""}', 'Response Time (minutes)', 'Frequency')
    return plot_to_base64(plt)

def analyze_sentiment_over_time(df, username=None):
    """Analyze sentiment over time and return base64 image of the plot."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
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
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()
    
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    # Define a function to categorize sentiment into emotions
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
    
    # Apply sentiment analysis and emotion categorization
    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
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
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['emojis'] = df_filtered['message'].apply(extract_emojis)
    all_emojis = [emoji for sublist in df_filtered['emojis'] for emoji in sublist]
    top_emojis = Counter(all_emojis).most_common(5)

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
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Calculate sentiment polarity and subjectivity
    df_filtered['polarity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_filtered['subjectivity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    # Count the number of messages for each sentiment
    sentiment_counts = df_filtered.groupby(['polarity', 'subjectivity']).size().reset_index(name='counts')

    plt.figure(figsize=(10, 8))
    plt.scatter(sentiment_counts['polarity'], sentiment_counts['subjectivity'], s=sentiment_counts['counts']*10, alpha=0.6, color='purple')
    apply_consistent_plot_styling(plt, f'Sentiment Distribution {"for " + username if username else ""}', 'Polarity (Positive/Negative)', 'Subjectivity (Objective/Subjective)')
    return plot_to_base64(plt)

def plot_vocabulary_diversity(df, username=None):
    """
    Plot a scatter plot of vocabulary diversity over time and return base64 image.
    x-axis: Unique words used
    y-axis: Average message length
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()
        
    df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower() # optimization for unique words
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    word_matrix = vectorizer.fit_transform(df_filtered['clean_message_lower'].dropna())
    unique_words_count = len(vectorizer.get_feature_names_out())

    # Calculate average message length
    df_filtered['avg_message_length'] = df_filtered['message'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 8))
    plt.scatter(unique_words_count, df_filtered['avg_message_length'].mean(), color='green')
    apply_consistent_plot_styling(plt, f'Vocabulary Diversity {"for " + username if username else ""}', 'Unique Words (Avg)', 'Message Length (Avg)')
    return plot_to_base64(plt)

def plot_language_complexity_pos(df, username=None):
    """
    Analyze and plot the distribution of POS tags for a user or the entire chat,
    and return a base64 image of the plot.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Function to extract POS tags from a message
    def extract_pos_tags(message):
        analysis = TextBlob(message)
        return [tag for (word, tag) in analysis.tags]

    # Apply POS tag extraction to each message
    df_filtered['pos_tags'] = df_filtered['message'].apply(extract_pos_tags)

    # Flatten the list of POS tags and count their occurrences
    all_pos_tags = [tag for sublist in df_filtered['pos_tags'] for tag in sublist]
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

    # Analyze interactions and add edges
    for i in range(len(df) - 1):
        sender = df['name'].iloc[i]
        next_sender = df['name'].iloc[i + 1]
        if sender != next_sender:
            if G.has_edge(sender, next_sender):
                G[sender][next_sender]['weight'] += 1
            else:
                G.add_edge(sender, next_sender, weight=1)

    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=8, width=[d['weight'] / 5 for u, v, d in G.edges(data=True)])
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("User Relationship Graph", fontsize=15)
    #plt.tight_layout()

    return plot_to_base64(plt)

def plot_skills_radar_chart(df, username=None):
    """
    Generate a radar chart to visualize various skills based on keyword analysis.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    

    # Count keyword occurrences for each skill
    skill_counts = {}
    for skill, keywords in skill_keywords.items():
        skill_counts[skill] = sum(df_filtered['clean_message'].str.lower().str.count('|'.join(keywords)))

    # Prepare data for radar chart
    skills = list(skill_counts.keys())
    counts = list(skill_counts.values())
    
    # Normalize counts for radar chart
    max_count = max(counts)
    normalized_counts = [(count / (max_count+1)) for count in counts]

    # Number of variables (skills)
    num_vars = len(skills)

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
    #plt.tight_layout()

    return plot_to_base64(plt)