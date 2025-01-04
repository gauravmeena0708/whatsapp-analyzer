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
nltk.download('stopwords')
from collections import Counter

class GroupAnalyzer:
    URL_PATTERN = r"(https?://\S+)"
    YOUTUBE_PATTERN = r"(https?://youtu(\.be|be\.com)\S+)"

    def __init__(self, file_path):
        self.file_path = file_path
        self.youtube_pattern = self.YOUTUBE_PATTERN
        self.url_pattern = self.URL_PATTERN

    def parse_chat_data(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            chat_lines = file.readlines()

        chat_data = []
        for line in chat_lines:
            try:
                match = re.match(
                    r"(\d{2}\/\d{2}\/\d{2}, \d{1,2}:\d{2} [apm]{2}) - (.*?): (.*)", line
                )
                if match:
                    timestamp, sender, message = match.groups()
                    # More robust date/time parsing
                    date_obj = parser.parse(timestamp)
                    chat_data.append({"t": date_obj, "name": sender, "message": message})
            except (ValueError, AttributeError):  # Catch potential errors during parsing
                print(f"Skipping line: {line.strip()} (Parse error)")

        return pd.DataFrame(chat_data)

    # ... (chunk_column - if needed, otherwise remove) ...

    def generate_wordcloud(self, text, stop_words=None):
        text = re.sub(r"<Media omitted>", "", text)
        text = re.sub(r"https", "", text)

        if stop_words is None:
            stop_words = set(stopwords.words("english"))
            stop_words.update(["omitted", "media"])
            # Add more custom stop words if needed

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
        plt.show()

    def get_emojis(self, text):
        emoji_list = []
        data = regex.findall(r"\X", text)
        for word in data:
            if any(char in emoji.EMOJI_DATA for char in word):
                emoji_list.append(word)
        return emoji_list

    def get_urls(self, text):
        url_list = regex.findall(self.url_pattern, text)
        return url_list

    def get_yturls(self, text):
        url_list = re.findall(self.youtube_pattern, text)
        return url_list

    def df_basic_cleanup(self, df):
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

        df["emoji"] = df["message"].apply(self.get_emojis)
        df["emojicount"] = df["emoji"].str.len()

        df["urls"] = df["message"].apply(self.get_urls)
        df["urlcount"] = df["urls"].str.len()

        df["yturls"] = df["message"].apply(self.get_yturls)
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

    def create_plotly_fig(self, data, x, y, sortby, asc=False, count=True):
        if count:
            grouped_data = data.groupby(x, as_index=False)[y].count()
        else:
            grouped_data = data.groupby(x, as_index=False)[y].sum()
        if sortby != 0:
            grouped_data = grouped_data.sort_values(sortby, ascending=asc)

        fig = px.line(
            data_frame=grouped_data,
            x=x,
            y=y,
            title=f"Number of {y} by {x}",
            labels={x: x, y: f"Number of {y}"},
        )
        # Show the figure.
        return fig  # .show()

    def calculate_word_frequency(self, df, column_name="message"):
        """Calculates the frequency of words in a specified column."""
        stop_words = set(stopwords.words("english"))
        word_counts = Counter()

        for message in df[column_name]:
            words = re.findall(r"\b\w+\b", message.lower())  # Extract words
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_counts[word] += 1

        return word_counts


def main():
    """
    Analyzes a WhatsApp chat file using the GroupAnalyzer class.
    """
    chat_file = "whatsapp_chat.txt"  # Replace with your chat file path

    try:
        # Create an instance of GroupAnalyzer
        analyzer = GroupAnalyzer(chat_file)

        # Parse the chat data
        df = analyzer.parse_chat_data()

        # Perform basic cleanup and feature engineering
        df = analyzer.df_basic_cleanup(df)

        # Example analyses (uncomment and modify as needed):

        # 1. Generate a word cloud of the messages
        # analyzer.generate_wordcloud(df["message"].str.cat(sep=" "))

        # 2. Get the top 10 most frequent words
        word_frequencies = analyzer.calculate_word_frequency(df)
        top_10_words = word_frequencies.most_common(10)
        print("Top 10 most frequent words:")
        for word, count in top_10_words:
            print(f"{word}: {count}")

        # 3. Create a Plotly chart of message count by day of the week

        #fig = analyzer.create_plotly_fig(df, x="dayn", y="message", sortby="dow")
        #fig.show()  # Or save the figure using fig.write_html("output.html")

        # 4. Analyze the distribution of message lengths
        print("\nMessage Length Distribution:")
        print(df["mlen"].describe())

        # 5. Count the total number of media messages
        total_media = df["mediacount"].sum()
        print(f"\nTotal Media Messages: {total_media}")

        # 6. Analyze emoji usage (count, most frequent, etc.)
        total_emojis = df["emojicount"].sum()
        print(f"\nTotal Emojis Used: {total_emojis}")

        # Example: Get the 5 most frequent emojis
        all_emojis = [e for sublist in df["emoji"] for e in sublist]
        emoji_counts = pd.Series(all_emojis).value_counts()
        top_5_emojis = emoji_counts.head(5)
        print("\nTop 5 Most Frequent Emojis:")
        print(top_5_emojis)

        # ... Add more analysis as needed ...

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()