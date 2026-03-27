# whatsapp_analyzer/plot_utils.py
import base64

import json
import uuid

def wrap_base64_img(img_base64):
    return f'<img src="data:image/png;base64,{img_base64}" alt="Plot" style="max-width: 100%; height: auto; border-radius: 8px;">'

def render_chartjs(config):
    chart_id = "chart_" + uuid.uuid4().hex[:8]
    html = f'''
    <div style="position: relative; height: 300px; width: 100%;">
        <canvas id="{chart_id}"></canvas>
    </div>
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {json.dumps(config)});
    }});
    </script>
    '''
    return html
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
import numpy as np
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
    try:
        from .ml_models import predict_sentiment
        return predict_sentiment(text)
    except Exception:
        pass
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

def plot_to_base64(plt, wrap=True):
    """Convert a Matplotlib plot to a base64 encoded image."""
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close()
    return wrap_base64_img(img_base64) if wrap else img_base64

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
    df_filtered = _filter_by_user(df, username)
    if 'sentiment' not in df_filtered.columns:
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])

    counts, bins = np.histogram(df_filtered['sentiment'].dropna(), bins=20)
    labels = [f"{bins[i]:.2f} to {bins[i+1]:.2f}" for i in range(len(counts))]
    data = [int(x) for x in counts]
    
    title = f'Sentiment Distribution {"for " + username if username else ""}'
    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Frequency", "data": data, "backgroundColor": "#87CEEB"}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "Sentiment Polarity"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Frequency"}}
            }
        }
    }
    return render_chartjs(config)

def plot_most_active_hours(df, username=None):
    df_filtered = _filter_by_user(df, username)
    message_counts_by_hour = df_filtered['hour'].value_counts().sort_index()

    labels = [str(x) for x in message_counts_by_hour.index]
    data = [int(x) for x in message_counts_by_hour.values]
    title = f'Most Active Hours {"for " + username if username else ""}'
    
    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Messages", "data": data, "backgroundColor": "#87CEEB"}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "Hour of the Day"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Number of Messages"}}
            }
        }
    }
    return render_chartjs(config)

def generate_wordcloud(df, username=None):
    """Generate word cloud and return base64 image."""
    df_filtered = _filter_by_user(df, username)

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    text = " ".join(msg for msg in df_filtered['clean_message'] if isinstance(msg, str) and len(msg.strip())>0)

    # Strip non-ASCII characters (emojis, Devanagari, etc.) — WordCloud's default
    # font (DroidSansMono) cannot render them, causing rectangular boxes (tofu).
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    plt.figure(figsize=(10, 8))
    if not text.strip():
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
    if response_times is None or len(response_times) == 0:
        return "<p class='text-center text-muted'>Not enough messages to compute response times.</p>"

    counts, bins = np.histogram(response_times.dropna(), bins=20)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(counts))]
    data = [int(x) for x in counts]
    
    title = f'Response Time Distribution {"for " + username if username else ""}'
    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Frequency", "data": data, "backgroundColor": "#87CEEB"}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "Response Time (minutes)"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Frequency"}}
            }
        }
    }
    return render_chartjs(config)

def analyze_sentiment_over_time(df, username=None):
    df_filtered = _filter_by_user(df, username)
    if df_filtered.empty:
        return "<p class='text-center text-muted'>No messages available for sentiment trend analysis.</p>"

    if 'sentiment' not in df_filtered.columns:
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    daily_sentiment = df_filtered['sentiment'].resample('W').mean().dropna()
    
    labels = [x.strftime('%Y-%m-%d') for x in daily_sentiment.index]
    data = [float(x) for x in daily_sentiment.values]

    title = f'Sentiment Over Time {"for " + username if username else ""}'
    config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Average Sentiment", "data": data, "borderColor": "purple", "fill": False, "tension": 0.1}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "Date"}},
                "y": {"title": {"display": True, "text": "Average Sentiment"}}
            }
        }
    }
    return render_chartjs(config)

def analyze_emotion_over_time(df, username=None):
    df_filtered = _filter_by_user(df, username)
    if df_filtered.empty:
        return "<p class='text-center text-muted'>No messages available for emotion trend analysis.</p>"

    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    if 'sentiment' not in df_filtered.columns:
        df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: _polarity_subjectivity(x)[0])

    def categorize_emotion(score):
        if score > 0.5: return "joy"
        elif score > 0: return "surprise"
        elif score < -0.5: return "sadness"
        elif score < 0: return "anger"
        else: return "neutral"

    df_filtered['emotion'] = df_filtered['sentiment'].apply(categorize_emotion)
    daily_emotions = df_filtered.groupby(pd.Grouper(freq='D'))['emotion'].apply(lambda x: x.value_counts()).unstack(fill_value=0).resample('W').sum()

    labels = [x.strftime('%Y-%m-%d') for x in daily_emotions.index]
    datasets = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
    for idx, emotion in enumerate(daily_emotions.columns):
        datasets.append({
            "label": emotion,
            "data": [int(x) for x in daily_emotions[emotion].values],
            "borderColor": colors[idx % len(colors)],
            "fill": False,
            "tension": 0.1
        })
        
    title = f'Emotion Trends Over Time {"for " + username if username else ""}'
    config = {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "Date"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Emotion Score"}}
            }
        }
    }
    return render_chartjs(config)

def plot_emoji_usage(df, username=None):
    df_filtered = _filter_by_user(df, username)
    df_filtered['emojis'] = df_filtered['message'].apply(extract_emojis)
    all_emojis = [emoji for sublist in df_filtered['emojis'] for emoji in sublist]
    top_emojis = Counter(all_emojis).most_common(5)

    if not top_emojis:
        return "<p class='text-center text-muted' style='margin-top: 50px;'>No emojis found.</p>"

    emojis, counts = zip(*top_emojis)
    labels = list(emojis)
    data = list(counts)
    
    title = f'Emoji Usage {"for " + username if username else ""}'
    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Count", "data": data, "backgroundColor": "#87CEEB"}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "Emoji"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Count"}}
            }
        }
    }
    return render_chartjs(config)

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
    
    title = f'Vocabulary Diversity {"for " + username if username else ""}'
    config = {
        "type": "bar",
        "data": {
            "labels": ["Unique Words", "Total Words"],
            "datasets": [{
                "label": "Word Count",
                "data": [unique_words_count, total_words_count],
                "backgroundColor": ["#075e54", "#25d366"]
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": False}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Word Count"}}
            }
        }
    }
    return render_chartjs(config)

def plot_language_complexity_pos(df, username=None):
    df_filtered = _filter_by_user(df, username)
    def extract_pos_tags(message):
        if TextBlob is None:
            return []
        analysis = TextBlob(message)
        return [tag for (word, tag) in analysis.tags]

    df_filtered['pos_tags'] = df_filtered['message'].apply(extract_pos_tags)
    all_pos_tags = [tag for sublist in df_filtered['pos_tags'] for tag in sublist]
    if not all_pos_tags:
        return "<p class='text-center text-muted'>No POS tags available for this dataset.</p>"

    pos_counts = Counter(all_pos_tags)
    labels = list(pos_counts.keys())
    data = list(pos_counts.values())

    title = f'POS Tag Distribution {"for " + username if username else ""}'
    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Count", "data": data, "backgroundColor": "#87CEEB"}]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "x": {"title": {"display": True, "text": "POS Tag"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Count"}}
            }
        }
    }
    return render_chartjs(config)

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

def plot_skills_radar_chart(skill_counts, username=None):
    skills = list(skill_counts.keys())
    counts = list(skill_counts.values())
    if counts and max(counts) > 0:
        max_val = max(counts)
        normalized_counts = [c / max_val for c in counts]
    else:
        return "<p class='text-center text-muted' style='margin-top: 50px;'>No skill categories available.</p>"

    title = f'Skills Radar Chart {"for " + username if username else ""}'
    config = {
        "type": "radar",
        "data": {
            "labels": skills,
            "datasets": [{
                "label": "Skill Level",
                "data": normalized_counts,
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgb(54, 162, 235)",
                "pointBackgroundColor": "rgb(54, 162, 235)",
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": "rgb(54, 162, 235)"
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "r": {"angleLines": {"display": True}, "suggestedMin": 0, "suggestedMax": 1}
            }
        }
    }
    return render_chartjs(config)

def plot_personality_radar(ocean_scores, username=None):
    if not ocean_scores:
        return "<p class='text-center text-muted' style='margin-top: 50px;'>Personality data unavailable (need more messages).</p>"

    traits = list(ocean_scores.keys())
    values = list(ocean_scores.values())

    title = f'Big Five Personality Traits {"for " + username if username else ""}'
    config = {
        "type": "radar",
        "data": {
            "labels": traits,
            "datasets": [{
                "label": "Percentile",
                "data": values,
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "borderColor": "rgb(255, 99, 132)",
                "pointBackgroundColor": "rgb(255, 99, 132)",
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": "rgb(255, 99, 132)"
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "r": {"angleLines": {"display": True}, "suggestedMin": 0, "suggestedMax": 1, "ticks": {"display": False}}
            }
        }
    }
    return render_chartjs(config)

def plot_interaction_matrix(matrix, username=None):
    if not matrix:
        return "<p class='text-center text-muted'>Interaction data unavailable.</p>"

    users = list(matrix.keys())
    if not users: return ""
    
    # If username is provided, we might want to highlight their row/col
    # but for simplicity we render the full matrix or a slice
    
    labels = users
    datasets = []
    
    # Chart.js Heatmap is tricky, we'll use a bubble chart or a colored grid if possible.
    # Actually, a Bar chart (stacked or group) showing "Who I interact with most" is better for UI.
    
    if username and username in matrix:
        # Show who 'username' interacts with
        interactions = matrix[username]
        sorted_inter = sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:10]
        if not sorted_inter: return "<p class='text-center text-muted'>No interactions found.</p>"
        
        target_users, counts = zip(*sorted_inter)
        
        config = {
            "type": "bar",
            "data": {
                "labels": list(target_users),
                "datasets": [{"label": "Interactions", "data": list(counts), "backgroundColor": "#25d366"}]
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"title": {"display": True, "text": f"Top Interactions for {username}"}}
            }
        }
        return render_chartjs(config)
    
    return "<p class='text-center text-muted'>Global matrix visualization pending.</p>"