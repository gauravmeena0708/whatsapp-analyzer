# analyzer.py (inside whatsapp_analyzer)
import os
from .parser import Parser
from .utils import (
    df_basic_cleanup,
)
# analysis_utils contains basic_stats, which in turn uses plot_utils and other analysis functions.
# analyzer.py only needs to import basic_stats from analysis_utils.
from .analysis_utils import basic_stats
from .constants import html_template # Used directly in generate_report

# Unused imports that were previously kept "for now" or for indirect dependencies:
# from .plot_utils import (
#     clean_message,
#     extract_emojis,
#     apply_consistent_plot_styling,
#     plot_activity_heatmap,
#     plot_sentiment_distribution,
#     plot_most_active_hours,
#     generate_wordcloud,
#     analyze_language_complexity,
#     plot_response_time_distribution,
#     analyze_sentiment_over_time,
#     analyze_emotion_over_time,
#     plot_emoji_usage,
#     plot_sentiment_bubble,
#     plot_vocabulary_diversity,
#     plot_language_complexity_pos,
#     plot_user_relationship_graph,
#     plot_skills_radar_chart,
# )
# from .analysis_utils import (
#     analyze_behavioral_traits, # Used by basic_stats
#     generate_behavioral_insights_text, # Used by basic_stats
#     analyze_hindi_abuse, # Used by basic_stats
# )
# from .constants import (
#     custom_hinglish_stopwords, # Referenced in analysis_utils via stop_words
#     skill_keywords, # Referenced in analysis_utils
#     hindi_abusive_words, # Referenced in analysis_utils
#     stop_words # Referenced in analysis_utils
# )
# External library imports that are no longer directly used in this file:
# import warnings
# import pandas as pd
# from textblob import TextBlob
# import nltk
# from nltk.corpus import stopwords
# from collections import Counter
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# from wordcloud import WordCloud
# import emoji
# import base64
# from io import BytesIO
# from functools import lru_cache
# import networkx as nx
# import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt

class WhatsAppAnalyzer:
    def __init__(self, chat_file, out_dir="."):
        self.chat_file = chat_file
        self.out_dir = out_dir
        self.parser = Parser(self.chat_file)
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
                # Call basic_stats from analysis_utils
                user_stats = basic_stats(self.df, name) 
                
                # Generate HTML for top 5 emojis and add to user_stats
                # user_stats["Top 5 Emojis"] is a list of (emoji, count) tuples
                top_5_emojis_html_str = " ".join(
                    [f"{emoji_val} ({count})" for emoji_val, count in user_stats["Top 5 Emojis"]]
                )
                user_stats['top_5_emojis_html'] = top_5_emojis_html_str
                
                # The n-gram and abuse count formatting is now done in basic_stats.
                # html_template will directly use these pre-formatted HTML strings.
                # We can now use dictionary unpacking for formatting.
                final_html = html_template.format(name=name, **user_stats)

                output_path = os.path.join(
                    self.out_dir, f"{name}_report.html"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(final_html)

                print(f"Report for {name} has been generated and saved at {output_path}")
    
# Example usage (you can put this in a separate script or in your main function):
# if __name__ == "__main__":
#     analyzer = WhatsAppAnalyzer(chat_file="../data/whatsapp_chat.txt", out_dir="../data")
#     analyzer.generate_report()