# whatsapp_analyzer/analyzer.py
import re
import regex
import emoji
from datetime import datetime
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from calendar import day_name
import numpy as np
import plotly.express as px
from dateutil import parser  # For more robust date parsing
import nltk
from collections import Counter

# Assuming you have utility functions in a separate module
from whatsapp_analyzer import utils
from whatsapp_analyzer import exceptions

import re
import regex
import emoji
from datetime import datetime
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from calendar import day_name
import numpy as np
import plotly.express as px
from dateutil import parser
import nltk
from collections import Counter
import sys
import traceback

# Assuming you have utility functions in a separate module
from whatsapp_analyzer import utils
from whatsapp_analyzer import exceptions

class Analyzer:
    def __init__(self, chat_data):
        """
        Initializes the Analyzer with chat data (DataFrame).

        Args:
            chat_data (pd.DataFrame): DataFrame containing parsed chat data.
        """
        self.chat_data = chat_data

    def generate_wordcloud(self, column_name="message", stop_words=None):
        """Generates and displays a word cloud from the specified column."""
        text = self.chat_data[column_name].str.cat(sep=" ")
        text = re.sub(r"<Media omitted>", "", text)
        text = re.sub(r"https", "", text)  # Remove common words

        if stop_words is None:
            stop_words = set(stopwords.words("english"))
            stop_words.update(["omitted", "media"])

        try:
            wordcloud = WordCloud(
                width=1600,
                height=800,
                stopwords=stop_words,
                background_color="black",
                colormap="rainbow",
            ).generate(text)

            plt.figure(figsize=(32, 18))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            #plt.show()
        except ValueError as e:
            print(f"Error generating word cloud: {e}")

    def calculate_word_frequency(self, column_name="message"):
        """Calculates the frequency of words in a specified column."""
        stop_words = set(stopwords.words("english"))
        word_counts = Counter()

        for message in self.chat_data[column_name]:
            words = re.findall(r"\b\w+\b", message.lower())  # Extract words
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_counts[word] += 1

        return word_counts

    def analyze_message_length(self, column_name="mlen"):
        """Analyzes the distribution of message lengths."""
        print("\nMessage Length Distribution:")
        print(self.chat_data[column_name].describe())

    def analyze_message_length(self, column_name="mlen"):
        """Analyzes the distribution of message lengths."""
        print("\nMessage Length Distribution:")
        series = self.chat_data[column_name].describe()
        print(series)
        return series  # Now it returns the pandas Series

    def analyze_media_count(self, column_name="mediacount"):
        """Counts the total number of media messages."""
        total_media = self.chat_data[column_name].sum()
        print(f"\nTotal Media Messages: {total_media}")

    def analyze_emoji_usage(self, column_name="emojicount"):
        """Analyzes emoji usage (total count, most frequent)."""
        total_emojis = self.chat_data[column_name].sum()
        print(f"\nTotal Emojis Used: {total_emojis}")

        all_emojis = [e for sublist in self.chat_data["emoji"] for e in sublist]
        emoji_counts = pd.Series(all_emojis).value_counts()
        return emoji_counts

    # analyzer.py
    def create_plotly_fig(self, x, y, sortby, asc=False, count=True):
        print(self.chat_data.columns)
        print(self.chat_data['dow'].unique())
        print(x, y)
        """
        Creates a Plotly line chart for visualization.
        """
        try:
            if count:
                grouped_data = self.chat_data.groupby(x, as_index=False, observed=True)[y].count()
            else:
                grouped_data = self.chat_data.groupby(x, as_index=False, observed=True)[y].sum()
            if sortby != 0:
                grouped_data = grouped_data.sort_values(sortby, ascending=asc)
        except Exception as e:
            print("An error occurred during grouping:")
            print(f"  Error message: {e}")
            print(f"  Error type: {type(e).__name__}")
            print("  Traceback:")
            traceback.print_exc()  # Print the full traceback
            return None  # Or raise the exception again if you want it to propagate

        fig = px.line(
            data_frame=grouped_data,
            x=x,
            y=y,
            title=f"Number of {y} by {x}",
            labels={x: x, y: f"Number of {y}"},
        )
        return fig