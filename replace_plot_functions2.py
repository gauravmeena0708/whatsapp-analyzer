import re

with open('whatsapp_analyzer/plot_utils.py', 'r') as f:
    content = f.read()
    
# Import numpy at the top
if 'import numpy as np' not in content:
    content = content.replace('import pandas as pd', 'import pandas as pd\nimport numpy as np')

def replace_func(func_name, new_code):
    global content
    pattern = rf'def {func_name}\(.*?\):.*?(?=\n\n?def |\Z)'
    content = re.sub(pattern, new_code, content, flags=re.DOTALL)

# plot_sentiment_distribution
replace_func('plot_sentiment_distribution', '''def plot_sentiment_distribution(df, username=None):
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
    return render_chartjs(config)''')

# plot_response_time_distribution
replace_func('plot_response_time_distribution', '''def plot_response_time_distribution(response_times, username=None):
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
    return render_chartjs(config)''')

# plot_vocabulary_diversity
replace_func('plot_vocabulary_diversity', '''def plot_vocabulary_diversity(df, username=None):
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
    return render_chartjs(config)''')

# plot_language_complexity_pos
replace_func('plot_language_complexity_pos', '''def plot_language_complexity_pos(df, username=None):
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
    return render_chartjs(config)''')

with open('whatsapp_analyzer/plot_utils.py', 'w') as f:
    f.write(content)
