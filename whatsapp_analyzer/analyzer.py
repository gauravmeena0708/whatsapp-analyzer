import re
import emoji
from datetime import datetime
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Assuming you have utility functions in a separate module
from whatsapp_analyzer import utils

class Analyzer:
    def __init__(self, chat_data):
        """
        Initializes the Analyzer with chat data (DataFrame).

        Args:
            chat_data (pd.DataFrame): DataFrame containing parsed chat data.
        """
        self.chat_data = chat_data

    def calculate_num_users(self):
        # Get unique users from the 'name' column in the DataFrame
        if "name" not in self.chat_data.columns:
            raise ValueError("Column 'name' not found in chat_data.")
        return self.chat_data["name"].nunique()


    def calculate_chat_period(self):
        """
        Calculates the start and end dates of the chat data.

        Returns:
            tuple: Start date and end date as strings.
        """
        if "date_time" not in self.chat_data.columns:
            raise ValueError("Column 'date_time' not found in chat_data.")
        
        start_date = self.chat_data["date_time"].min()
        end_date = self.chat_data["date_time"].max()
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


    def calculate_top_users(self, top_n=5):
        """
        Calculates the top N users by message count.

        Args:
            top_n (int): Number of top users to return.

        Returns:
            pd.DataFrame: DataFrame containing the top N users and their message counts.
        """
        if "name" not in self.chat_data.columns:
            raise ValueError("Column 'name' not found in chat_data.")
        
        if "name" in self.chat_data.columns:
            self.chat_data = self.chat_data[self.chat_data["name"].notnull()]

        user_message_counts = self.chat_data["name"].value_counts().head(top_n)
        return user_message_counts.reset_index().rename(columns={"index": "name", "name": "message_count"})

    
    def generate_wordcloud(self, column_name="message", stop_words=None):
        """Generates and displays a word cloud from the specified column."""
        text = self.chat_data[column_name].str.cat(sep=" ")
        text = re.sub(r"<Media omitted>", "", text)
        text = re.sub(r"https", "", text)  # Remove common words

        if stop_words is None:
            stop_words = set(stopwords.words("english"))
            stop_words.update(["omitted", "media", "https"])


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
        series = self.chat_data[column_name].describe()
        return series  # Now it returns the pandas Series

    def analyze_media_count(self, column_name="mediacount"):
        return self.chat_data[column_name].sum() if column_name in self.chat_data else 0



    
    def analyze_emoji_usage(self, column_name="emoji"):
        if column_name in self.chat_data:
            all_emojis = [e for sublist in self.chat_data[column_name].dropna() for e in sublist]
            emoji_counts = pd.Series(all_emojis).value_counts()
            return emoji_counts
        return pd.Series(dtype="int64")


    def create_seaborn_fig(self, x, y, sortby=None, asc=False, count=True):
        """
        Creates a Seaborn line chart for visualization.
        
        Args:
            x (str): Column name for the x-axis (e.g., 'dow' for days of the week).
            y (str): Column name for the y-axis (e.g., 'message' or 'message count').
            sortby (str): Column to sort the data by.
            asc (bool): Whether to sort in ascending order.
            count (bool): If True, count occurrences; if False, sum the values.

        Returns:
            dict: Path to the saved plot image.
        """

        try:
            # Grouping the data
            if count:
                grouped_data = self.chat_data.groupby(x, as_index=False, observed=True)[y].count()
            else:
                grouped_data = self.chat_data.groupby(x, as_index=False, observed=True)[y].sum()

            # Sorting if necessary
            if sortby:
                grouped_data = grouped_data.sort_values(by=sortby, ascending=asc)

        except Exception as e:
            print(f"An error occurred during grouping or plotting: {e}")
            traceback.print_exc()  # Print the full traceback for debugging
            return None

        # Create Seaborn line plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=grouped_data, x=x, y=y, marker='o')
        plt.title(f"Number of {y} by {x}")
        plt.xlabel(x)
        plt.ylabel(f"Number of {y}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot as an image
        chart_path = f'./data/{x}_vs_{y}_line_plot.png'
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        # Return the image path for further use in reports or as a response
        return [{
            "type": "image",
            "data": chart_path,
            "width": 800,
            "height": 400,
        }]
