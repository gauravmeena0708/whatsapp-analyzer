# analyzer.py (inside whatsapp_analyzer)
import os
from .parser import Parser
from .utils import (
    df_basic_cleanup,
    clean_message,
    apply_consistent_plot_styling,
    plot_activity_heatmap,
    plot_sentiment_distribution,
    plot_most_active_hours,
    generate_wordcloud,
    analyze_language_complexity,
    analyze_message_timing,
    plot_response_time_distribution,
    analyze_sentiment_over_time,
    analyze_emotion_over_time,
    plot_emoji_usage,
    plot_sentiment_bubble,
    plot_vocabulary_diversity,
    plot_language_complexity_pos,
    plot_user_relationship_graph,
    plot_skills_radar_chart,
    extract_emojis
)
from .constants import (
    custom_hinglish_stopwords,
    skill_keywords,
    hindi_abusive_words,
    html_template,
    stop_words
)
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import emoji
import base64
from io import BytesIO
from functools import lru_cache
import networkx as nx
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings
import pandas as pd

class WhatsAppAnalyzer:
    def __init__(self, chat_file, out_dir="."):
        self.chat_file = chat_file
        self.out_dir = out_dir
        try:
            self.parser = Parser(self.chat_file)
        except FileNotFoundError:
            # Print a user-friendly message
            error_message = f"Error: The chat file was not found at the specified path: {self.chat_file}"
            print(error_message)
            # Optionally, re-raise the exception or exit
            # For this task, let's make it exit so the user sees the message clearly.
            raise SystemExit(error_message)
        self.df = df_basic_cleanup(self.parser.parse_chat_data())

    def generate_report(self, users=None):
        """
        Generates HTML reports for specified users.

        Args:
            users (list, optional): A list of usernames for which to generate reports. 
                                   If None, reports are generated for all users. Defaults to None.
        """
        if users is None:
            users = self.df["name"].unique()

        for name in users:
            if name != "System":
                user_stats = self.basic_stats(self.df, name)
                top_5_emojis_html = " ".join(
                    [f"{emoji} ({count})" for emoji, count in user_stats["Top 5 Emojis"]]
                )

                final_html = html_template.format(
                    name=name,
                    total_messages=user_stats["Total Messages"],
                    total_words=user_stats["Total Words"],
                    unique_users=user_stats["Unique Users"],
                    total_emojis=user_stats["Total Emojis"],
                    top_5_emojis=top_5_emojis_html,
                    total_urls=user_stats["Total URLs"],
                    total_youtube_urls=user_stats["Total YouTube URLs"],
                    total_media=user_stats["Total Media"],
                    total_edits=user_stats["Total Edits"],
                    total_deletes=user_stats["Total Deletes"],
                    average_message_length=user_stats["Average Message Length"],
                    average_sentence_length=user_stats["Average Sentence Length"],
                    positive_messages=user_stats["Positive Messages"],
                    negative_messages=user_stats["Negative Messages"],
                    morning_messages=user_stats["Morning Messages"],
                    midday_messages=user_stats["Mid-day Messages"],
                    evening_messages=user_stats["Evening Messages"],
                    night_messages=user_stats["Night Messages"],
                    most_active_period=user_stats["Most Active Period"],
                    unique_words_count=user_stats["Unique Words Count"],
                    common_unigrams="".join(
                        [
                            f"<li>{word[0]}: {word[1]}</li>"
                            for word in user_stats["Common Unigrams"]
                        ]
                    ),
                    common_bigrams="".join(
                        [
                            f"<li>{word[0]}: {word[1]}</li>"
                            for word in user_stats["Common Bigrams"]
                        ]
                    ),
                    common_trigrams="".join(
                        [
                            f"<li>{word[0]}: {word[1]}</li>"
                            for word in user_stats["Common Trigrams"]
                        ]
                    ),
                    average_response_time=user_stats["Average Response Time"],
                    activity_heatmap=user_stats["Activity Heatmap"],
                    sentiment_distribution=user_stats["Sentiment Distribution"],
                    word_cloud=user_stats["Word Cloud"],
                    language_complexity=user_stats["Language Complexity"],
                    response_time_distribution=user_stats["Response Time Distribution"],
                    sentiment_over_time=user_stats["Sentiment Over Time"],
                    emoji_usage=user_stats["Emoji Usage"],
                    sentiment_bubble=user_stats["Sentiment Bubble"],
                    vocabulary_diversity=user_stats["Vocabulary Diversity"],
                    language_complexity_pos=user_stats["Language Complexity POS"],
                    user_relationship_graph=user_stats["User Relationship Graph"],
                    skills_radar_chart=user_stats["Skills Radar Chart"],
                    behavioral_insights_text=user_stats["Behavioral Insights Text"],
                    emotion_over_time=user_stats["Emotion Over Time"],
                    hindi_abuse_count=user_stats["Hindi Abuse Counts"],
                    most_active_hours=user_stats["Most Active Hours"],
                )

                output_path = os.path.join(
                    self.out_dir, f"{name}_report.html"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(final_html)

                print(f"Report for {name} has been generated and saved at {output_path}")
    
    def analyze_behavioral_traits(self, df, username=None):
        """
        Analyze behavioral traits and return a dictionary of insights.
        """
        if username:
            df_filtered = df[df['name'] == username].copy()
        else:
            df_filtered = df.copy()

        traits = {}

        # --- Sentiment Analysis ---
        df_filtered['sentiment_polarity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df_filtered['sentiment_subjectivity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        traits['avg_sentiment_polarity'] = df_filtered['sentiment_polarity'].mean()
        traits['avg_sentiment_subjectivity'] = df_filtered['sentiment_subjectivity'].mean()

        # --- Psychometric Analysis ---
        traits['num_questions'] = df_filtered['message'].apply(lambda x: x.count('?')).sum()
        traits['num_exclamations'] = df_filtered['message'].apply(lambda x: x.count('!')).sum()
        traits['first_person_pronouns'] = df_filtered['clean_message'].str.lower().str.count(r'\b(i|me|my|mine|myself)\b').sum() 

        # --- Skill Analysis (Keyword-based) ---
        
        traits['skills'] = {}
        for skill, keywords in skill_keywords.items():
            traits['skills'][skill] = sum(df_filtered['clean_message'].str.lower().str.count('|'.join(keywords)))

        
        # --- Language Complexity ---
        df_filtered['sentence_length'] = df_filtered['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))) if str(x).strip() else 0)
        traits['avg_sentence_length'] = df_filtered['sentence_length'].mean()

        # --- Lexical Diversity ---
        df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower()
        vectorizer = CountVectorizer(stop_words=list(stop_words))
        word_matrix = vectorizer.fit_transform(df_filtered['clean_message_lower'].dropna())
        unique_words_count = len(vectorizer.get_feature_names_out())
        total_words_count = df_filtered['message'].apply(lambda x: len(str(x).split())).sum()
        traits['lexical_diversity'] = unique_words_count / total_words_count if total_words_count > 0 else 0

        return traits
    
    def generate_behavioral_insights_text(self, traits, most_active_period, avg_response_time):
        """
        Generate human-readable insights based on behavioral traits.
        """
        insights = []

        # Sentiment Hints
        if traits['avg_sentiment_polarity'] > 0.2:
            insights.append("Tends to express positive sentiment in messages.")
        elif traits['avg_sentiment_polarity'] < -0.2:
            insights.append("Tends to express negative sentiment in messages.")
        else:
            insights.append("Maintains a neutral tone in messages.")

        if traits['avg_sentiment_subjectivity'] > 0.5:
            insights.append("Expresses subjective opinions and evaluations.")
        else:
            insights.append("Tends to communicate more objectively.")

        # Psychometric Hints
        if traits['num_questions'] > 20:
            insights.append("Asks a lot of questions, possibly indicating curiosity or a need for clarification.")
        if traits['num_exclamations'] > 5:
            insights.append("Uses exclamations frequently, suggesting excitement or strong opinions.")
        if traits['first_person_pronouns'] > 10:
            insights.append("Often refers to themselves, which might indicate a focus on personal experiences or opinions.")

        # Skill Hints
        if traits['skills']['communication'] > 5:
            insights.append("Demonstrates strong communication skills based on keyword analysis.")
        if traits['skills']['technical'] > 5:
            insights.append("Exhibits technical skills based on keyword analysis.")
        if traits['skills']['leadership'] > 2:
            insights.append("Shows potential leadership qualities based on keyword analysis.")
        if traits['skills']['problem_solving'] > 5:
            insights.append("Appears to have good problem-solving skills based on keyword analysis.")
        if traits['skills']['teamwork'] > 5:
            insights.append("Likely a good team player based on keyword analysis.")

        # Timing Hints
        if avg_response_time is not None:
            if avg_response_time < 60:
                insights.append("Responds quickly to messages, indicating high engagement.")
            elif avg_response_time > 180:
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
            else:
                insights.append("Most active at night.")
        
        # Language Complexity Hints
        if traits['avg_sentence_length'] > 3:
            insights.append("Uses long and complex sentences.")
        else:
            insights.append("Uses short and concise sentences.")

        # Lexical Diversity Hints
        if traits['lexical_diversity'] > 0.7:
            insights.append("Exhibits high lexical diversity, indicating a broad vocabulary.")
        elif traits['lexical_diversity'] < 0.4:
            insights.append("Has low lexical diversity, suggesting a more repetitive or focused communication style.")
        else:
            insights.append("Shows moderate lexical diversity.")

        return "<br/>".join(insights)

    def analyze_hindi_abuse(self, df, username=None):
        """
        Analyze the use of Hindi abusive words and return a dictionary of counts.
        """
        if username:
            df_filtered = df[df['name'] == username].copy()
        else:
            df_filtered = df.copy()

        # Count occurrences of each abusive word
        abuse_counts = {}
        for word in hindi_abusive_words:
            count = df_filtered['clean_message'].str.lower().str.count(word).sum()
            if count > 1:  # Only include if count is greater than 1
                abuse_counts[word] = count

        return abuse_counts

    def basic_stats(self, df, username=None):
        """
        Calculate basic statistics about messages, including sentiment, time analysis,
        most common n-grams (unigrams, bigrams, trigrams), most active period, and visualizations.
        """
        if username:
            df_filtered = df[df['name'] == username].copy()
        else:
            df_filtered = df.copy()

        # Sentiment Analysis
        sentiments = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        positive_msgs = sum(sentiments > 0)
        negative_msgs = sum(sentiments < 0)

        # Time of Day Analysis
        def categorize_time_of_day(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 16:
                return 'Mid-day'
            elif 16 <= hour < 18:
                return 'Evening'
            else:
                return 'Night'

        df_filtered['time_of_day'] = df_filtered['hour'].apply(categorize_time_of_day)
        morning_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Morning'])
        midday_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Mid-day'])
        evening_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Evening'])
        night_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Night'])
        message_counts_by_period = {'Morning': morning_msgs, 'Mid-day': midday_msgs, 'Evening': evening_msgs, 'Night': night_msgs}
        most_active_period = max(message_counts_by_period, key=message_counts_by_period.get)

        # Clean messages
        df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))

        # Unique words count (optimized)
        df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower() # optimization for unique words
        vectorizer = CountVectorizer(stop_words=list(stop_words))
        word_matrix = vectorizer.fit_transform(df_filtered['clean_message_lower'].dropna())
        unique_words_count = len(vectorizer.get_feature_names_out())

        # Most Common unigrams, bigrams, trigrams (optimized)
        def get_top_ngrams(corpus, n=1, top_k=10):
            vec = CountVectorizer(ngram_range=(n, n), stop_words=list(stop_words)).fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq_filtered = [item for item in words_freq if item[1] > 1]
            words_freq = sorted(words_freq_filtered, key=lambda x: x[1], reverse=True)
            return words_freq[:top_k]
        
        common_unigrams = get_top_ngrams(df_filtered['clean_message_lower'].dropna(), 1, 10)
        common_bigrams = get_top_ngrams(df_filtered['clean_message_lower'].dropna(), 2, 10)
        common_trigrams = get_top_ngrams(df_filtered['clean_message_lower'].dropna(), 3, 10)

        # Top 5 Emojis
        df_filtered['emojis'] = df_filtered['message'].apply(extract_emojis)
        all_emojis = [emoji for sublist in df_filtered['emojis'] for emoji in sublist]
        top_5_emojis = Counter(all_emojis).most_common(5)

        # Average Sentence Length
        df_filtered['sentence_length'] = df_filtered['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))))
        avg_sentence_length = df_filtered['sentence_length'].apply(lambda x: len(str(x).split()) / x if x > 0 else 0).mean()

        # Analyze message timing and get response times
        response_times = analyze_message_timing(df, username)
        
        # Calculate average response time
        average_response_time = response_times.mean() if not response_times.empty else 0

        # Visualizations
        activity_heatmap_base64 = plot_activity_heatmap(df_filtered, username)
        sentiment_distribution_base64 = plot_sentiment_distribution(df_filtered, username)
        wordcloud_base64 = generate_wordcloud(df_filtered, username)
        language_complexity_base64 = analyze_language_complexity(df_filtered, username)
        response_time_distribution_base64 = plot_response_time_distribution(response_times, username)
        sentiment_over_time_base64 = analyze_sentiment_over_time(df, username)
        emoji_usage_base64 = plot_emoji_usage(df_filtered, username)
        sentiment_bubble_base64 = plot_sentiment_bubble(df_filtered, username)
        vocabulary_diversity_base64 = plot_vocabulary_diversity(df_filtered, username)
        language_complexity_pos_base64 = plot_language_complexity_pos(df_filtered, username)
        user_relationship_graph_base64 = plot_user_relationship_graph(df)
        skills_radar_chart_base64 = plot_skills_radar_chart(df_filtered, username)
        emotion_over_time_base64 = analyze_emotion_over_time(df_filtered, username)
        most_active_hours_base64 = plot_most_active_hours(df_filtered, username)
        # Analyze behavioral traits
        behavioral_traits = self.analyze_behavioral_traits(df_filtered, username)
        behavioral_insights_text = self.generate_behavioral_insights_text(behavioral_traits, most_active_period, average_response_time)

        # Analyze for Hindi गाली and get counts if count > 1
        abuse_counts = self.analyze_hindi_abuse(df_filtered, username)
        
        # Convert the abuse_counts dictionary to an HTML-formatted string
        abuse_counts_html = "<ul>"
        for word, count in abuse_counts.items():
            abuse_counts_html += f"<li>{word}: {count}</li>"
        abuse_counts_html += "</ul>"

        stats = {
            'Total Messages': len(df_filtered),
            'Total Words': df_filtered['message'].apply(lambda x: len(str(x).split())).sum(),
            'Unique Users': df_filtered['name'].nunique(),
            'Total Emojis': df_filtered['emojicount'].sum(),
            'Total URLs': df_filtered['urlcount'].sum(),
            'Total YouTube URLs': df_filtered['yturlcount'].sum(),
            'Total Media': df_filtered['mediacount'].sum(),
            'Total Edits': df_filtered['editcount'].sum(),
            'Total Deletes': df_filtered['deletecount'].sum(),
            'Average Message Length': df_filtered['message'].apply(lambda x: len(str(x).split())).mean(),
            'Positive Messages': positive_msgs,
            'Negative Messages': negative_msgs,
            'Morning Messages': morning_msgs,
            'Mid-day Messages': midday_msgs,
            'Evening Messages': evening_msgs,
            'Night Messages': night_msgs,
            'Most Active Period': most_active_period,
            'Unique Words Count': unique_words_count,
            'Common Unigrams': common_unigrams,
            'Common Bigrams': common_bigrams,
            'Common Trigrams': common_trigrams,
            'Top 5 Emojis': top_5_emojis,
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
            'Behavioral Traits': behavioral_traits,
            'Emotion Over Time': emotion_over_time_base64,
            'Behavioral Insights Text': behavioral_insights_text,
            'Hindi Abuse Counts': abuse_counts_html,
            'Most Active Hours': most_active_hours_base64,
        }

        return stats

# Example usage (you can put this in a separate script or in your main function):
# if __name__ == "__main__":
#     analyzer = WhatsAppAnalyzer(chat_file="../data/whatsapp_chat.txt", out_dir="../data")
#     analyzer.generate_report()