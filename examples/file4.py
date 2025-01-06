# usage.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from whatsapp_analyzer.parser import Parser
from whatsapp_analyzer import utils
from whatsapp_analyzer.analyzer import Analyzer
from whatsapp_analyzer.pdf_generator import PDFGenerator  # Import PDFGenerator from the package
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

def create_chat_summary_section(analyzer):
    """
    Generates a section with basic chat summary, including number of users,
    chat period, and top users based on message count.
    """
    num_users = analyzer.calculate_num_users()
    chat_period = analyzer.calculate_chat_period()
    top_users = analyzer.calculate_top_users(5)
    top_users_df = pd.DataFrame(top_users, columns=["User", "Message Count"])

    return [
        {"type": "html", "data": "<h3>Chat Summary</h3>"},
        {"type": "html", "data": f"<p><strong>Number of Users:</strong> {num_users}</p>"},
        {"type": "html", "data": f"<p><strong>Chat Period:</strong> {chat_period}</p>"},
        {"type": "html", "data": "<h4>Top 5 Users by Message Count</h4>"},
        {"type": "table", "data": top_users_df},
    ]

def main():
    chat_file = "./data/whatsapp_chat2.txt"
    wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    html_template_path = 'templates/template.html'
    output_pdf_path = './data/whatsapp_report.pdf'

    try:
        parser = Parser(chat_file)
        df = parser.parse_chat_data()
        df = utils.df_basic_cleanup(df)
        analyzer = Analyzer(df)

        # Use PDFGenerator from whatsapp_analyzer
        pdf_generator = PDFGenerator(html_template_path, output_pdf_path, wkhtmltopdf_path)

        content = [
            {"type": "html", "data": "<h2>WhatsApp Chat Analysis</h2>"},
            *create_chat_summary_section(analyzer),
            *create_word_cloud_section(analyzer),
            *create_top_10_words_section(analyzer),
            *create_message_length_section(analyzer),
            *create_media_count_section(analyzer),
            *create_top_5_emojis_section(analyzer),
            *create_seaborn_chart_section(analyzer)
        ]

        pdf_generator.generate_pdf(content, wkhtmltopdf_path=wkhtmltopdf_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
