import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from .parser import Parser  # Relative import for modules within the package
from .utils import df_basic_cleanup, clean_message, apply_consistent_plot_styling, plot_activity_heatmap, plot_sentiment_distribution, plot_most_active_hours, generate_wordcloud, analyze_language_complexity, analyze_message_timing, plot_response_time_distribution, analyze_sentiment_over_time, analyze_emotion_over_time, plot_emoji_usage, plot_sentiment_bubble, plot_vocabulary_diversity, plot_language_complexity_pos, plot_user_relationship_graph, plot_skills_radar_chart
from .analyzer import analyze_behavioral_traits, generate_behavioral_insights_text, analyze_hindi_abuse, basic_stats
from .constants import custom_hinglish_stopwords, skill_keywords, hindi_abusive_words, html_template
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

# Get all installed font names
available_fonts = {fm.FontProperties(fname=fp).get_name() for fp in fm.findSystemFonts()}

# Add an emoji-compatible font if available
emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji"]
selected_font = None

for font in emoji_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams["font.family"] = [selected_font, "Roboto", "DejaVu Sans", "sans-serif"]
else:
    warnings.warn(
        "No emoji-compatible font found. Install 'Segoe UI Emoji', or 'Apple Color Emoji' for full emoji support."
    )
    plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans", "sans-serif"]

def main():
    """Main function to run the WhatsApp chat analysis."""

    # Load and clean data
    chat_file = "../data/whatsapp_chat.txt"  # Replace with your chat file path
    parser = Parser(chat_file)
    df = df_basic_cleanup(parser.parse_chat_data())

    # Combine NLTK stopwords with custom Hinglish stopwords
    stop_words = set(stopwords.words('english')).union(custom_hinglish_stopwords)

    # Generate HTML report for each user
    for name in df['name'].unique():
        user_stats = basic_stats(df, name)
        top_5_emojis_html = " ".join([f"{emoji} ({count})" for emoji, count in user_stats['Top 5 Emojis']])

        final_html = html_template.format(
            name=name,
            total_messages=user_stats['Total Messages'],
            total_words=user_stats['Total Words'],
            unique_users=user_stats['Unique Users'],
            total_emojis=user_stats['Total Emojis'],
            top_5_emojis=top_5_emojis_html,
            total_urls=user_stats['Total URLs'],
            total_youtube_urls=user_stats['Total YouTube URLs'],
            total_media=user_stats['Total Media'],
            total_edits=user_stats['Total Edits'],
            total_deletes=user_stats['Total Deletes'],
            average_message_length=user_stats['Average Message Length'],
            average_sentence_length=user_stats['Average Sentence Length'],
            positive_messages=user_stats['Positive Messages'],
            negative_messages=user_stats['Negative Messages'],
            morning_messages=user_stats['Morning Messages'],
            midday_messages=user_stats['Mid-day Messages'],
            evening_messages=user_stats['Evening Messages'],
            night_messages=user_stats['Night Messages'],
            most_active_period=user_stats['Most Active Period'],
            unique_words_count=user_stats['Unique Words Count'],
            common_unigrams="".join([f"<li>{word[0]}: {word[1]}</li>" for word in user_stats['Common Unigrams']]),
            common_bigrams="".join([f"<li>{word[0]}: {word[1]}</li>" for word in user_stats['Common Bigrams']]),
            common_trigrams="".join([f"<li>{word[0]}: {word[1]}</li>" for word in user_stats['Common Trigrams']]),
            average_response_time=user_stats['Average Response Time'],
            activity_heatmap=user_stats['Activity Heatmap'],
            sentiment_distribution=user_stats['Sentiment Distribution'],
            word_cloud=user_stats['Word Cloud'],
            language_complexity=user_stats['Language Complexity'],
            response_time_distribution=user_stats['Response Time Distribution'],
            sentiment_over_time=user_stats['Sentiment Over Time'],
            emoji_usage=user_stats['Emoji Usage'],
            sentiment_bubble=user_stats['Sentiment Bubble'],
            vocabulary_diversity=user_stats['Vocabulary Diversity'],
            language_complexity_pos=user_stats['Language Complexity POS'],
            user_relationship_graph=user_stats['User Relationship Graph'],
            skills_radar_chart=user_stats['Skills Radar Chart'],
            behavioral_insights_text=user_stats['Behavioral Insights Text'],
            emotion_over_time=user_stats['Emotion Over Time'],
            hindi_abuse_count=user_stats['Hindi Abuse Counts'],
            most_active_hours=user_stats['Most Active Hours'],
        )

        output_path = os.path.join('..','data', f"{name}_report.html")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(final_html)

        print(f"Report for {name} has been generated and saved at {output_path}")

if __name__ == "__main__":
    main()