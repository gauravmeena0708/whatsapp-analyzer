import re

with open('whatsapp_analyzer/plot_utils.py', 'r') as f:
    content = f.read()

def replace_func(func_name, new_code):
    global content
    pattern = rf'def {func_name}\(.*?\):.*?(?=\n\n?def |\Z)'
    content = re.sub(pattern, new_code, content, flags=re.DOTALL)

# plot_most_active_hours
replace_func('plot_most_active_hours', '''def plot_most_active_hours(df, username=None):
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
    return render_chartjs(config)''')


# analyze_sentiment_over_time
replace_func('analyze_sentiment_over_time', '''def analyze_sentiment_over_time(df, username=None):
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
    return render_chartjs(config)''')


# analyze_emotion_over_time
replace_func('analyze_emotion_over_time', '''def analyze_emotion_over_time(df, username=None):
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
    return render_chartjs(config)''')


# plot_emoji_usage
replace_func('plot_emoji_usage', '''def plot_emoji_usage(df, username=None):
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
    return render_chartjs(config)''')


# plot_skills_radar_chart
replace_func('plot_skills_radar_chart', '''def plot_skills_radar_chart(df, username=None):
    df_filtered = _filter_by_user(df, username)
    skill_counts = {}
    for skill, keywords in skill_keywords.items():
        pattern = '|'.join(re.escape(kw) for kw in keywords)
        skill_counts[skill] = int(df_filtered['clean_message'].str.lower().str.count(pattern).sum())

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
    return render_chartjs(config)''')

with open('whatsapp_analyzer/plot_utils.py', 'w') as f:
    f.write(content)
