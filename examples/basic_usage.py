# examples/basic_usage.py
import re
from whatsapp_analyzer.parser import Parser
from whatsapp_analyzer import utils
from whatsapp_analyzer.analyzer import Analyzer
from whatsapp_analyzer.exceptions import ParseError
import plotly.offline as pyo

def main():
    """
    Example usage of the WhatsApp Analyzer.
    """
    chat_file = "./data/whatsapp_chat.txt"  # Replace with your chat file path

    try:
        # Parse the chat data
        parser = Parser(chat_file)
        df = parser.parse_chat_data()

        # Perform basic cleanup and feature engineering
        df = utils.df_basic_cleanup(df)  # Make sure this line is here\
        print(df.columns)

        # Create an Analyzer instance
        analyzer = Analyzer(df)

        # Example analyses:

        # 1. Generate a word cloud
        analyzer.generate_wordcloud()

        # 2. Get the top 10 most frequent words
        word_frequencies = analyzer.calculate_word_frequency()
        top_10_words = word_frequencies.most_common(10)
        print("\nTop 10 most frequent words:")
        for word, count in top_10_words:
            print(f"{word}: {count}")

        # 3. Create a Plotly chart of message count by day of the week
        #fig = analyzer.create_plotly_fig(x="dayn", y="message", sortby="dow")
        fig = analyzer.create_plotly_fig(x="dayn", y="message", sortby=0) # sortby is 0 so it won't sort
        pyo.plot(fig, filename='data/my_plot.html', auto_open=False)
        analyzer.analyze_message_length()

        # 5. Count the total number of media messages
        analyzer.analyze_media_count()

        # 6. Analyze emoji usage
        emoji_counts = analyzer.analyze_emoji_usage()
        top_5_emojis = emoji_counts.head(5)
        print("\nTop 5 Most Frequent Emojis:")
        print(top_5_emojis)

    except ParseError as pe:
        print("A parsing error occurred:")
        pe.print_exception_details()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()