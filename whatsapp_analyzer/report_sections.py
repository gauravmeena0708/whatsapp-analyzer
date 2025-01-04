# whatsapp_analyzer/report_sections.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import emoji

def create_word_cloud_section(analyzer):
    analyzer.generate_wordcloud()
    wordcloud_filename = os.path.abspath("./data/wordcloud.png")
    plt.savefig(wordcloud_filename)
    plt.close()
    return [{"type": "image", "data": wordcloud_filename, "width": 500, "height": 300}]

def create_top_10_words_section(analyzer):
    word_frequencies = analyzer.calculate_word_frequency()
    top_10_words = word_frequencies.most_common(10)
    top_10_words_df = pd.DataFrame(top_10_words, columns=["Word", "Count"])
    return [{"type": "html", "data": "<h3>Top 10 Most Frequent Words</h3>"}, {"type": "table", "data": top_10_words_df}]

def create_message_length_section(analyzer):
    message_length_stats = analyzer.analyze_message_length()
    message_length_df = message_length_stats.reset_index()
    message_length_df.columns = ["Stat", "Value"]
    return [{"type": "html", "data": "<h3>Message Length Distribution</h3>"}, {"type": "table", "data": message_length_df}]

def create_media_count_section(analyzer):
    media_count = analyzer.analyze_media_count()
    return [{"type": "html", "data": f"<h3>Total Media Messages: {media_count}</h3>"}]

def create_top_5_emojis_section(analyzer):
    emoji_counts = analyzer.analyze_emoji_usage()
    top_5_emojis = emoji_counts.head(5)
    top_5_emojis_df = pd.DataFrame(top_5_emojis.reset_index())
    top_5_emojis_df.columns = ["Emoji", "Count"]

    # Convert the DataFrame to HTML and apply the emoji class
    html_table = top_5_emojis_df.to_html(classes="table table-sm table-bordered border-primary d-print-table fs-6", index=False)

    # Apply emoji class to any column that contains emojis
    soup = BeautifulSoup(html_table, 'html.parser')
    for td in soup.find_all('td'):
        if emoji.emoji_count(td.string):  # Check if the text contains emojis
            td['class'] = td.get('class', []) + ['emoji']

    # Convert the modified HTML back to a string
    modified_html_table = str(soup)

    return [{"type": "html", "data": "<h3>Top 5 Most Frequent Emojis</h3>"}, {"type": "html", "data": modified_html_table}]

def create_seaborn_chart_section(analyzer):
    try:
        chart_data = analyzer.create_seaborn_fig(x="dow", y="message")  # Modify this according to your data
        chart_path = chart_data[0]["data"]  # Assuming the chart image is saved and path is returned
        return [{"type": "image", "data": chart_path, "width": 800, "height": 400}]
    except ValueError as e:
        print(f"Error: {e}")
        return [{"type": "html", "data": "<p>Error: Data not found. Unable to generate chart.</p>"}]
