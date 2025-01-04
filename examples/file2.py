import re
import os
import pdfkit
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bs4 import BeautifulSoup
from whatsapp_analyzer.parser import Parser
from whatsapp_analyzer import utils
from whatsapp_analyzer.analyzer import Analyzer
import traceback

class PDFGenerator:
    def __init__(self, html_template_path, output_path, wkhtmltopdf_path):
        self.template = Path(html_template_path).read_text()
        self.output_path = output_path
        self.wkhtmltopdf_path = wkhtmltopdf_path
        self.classes = 'table table-sm table-bordered border-primary d-print-table fs-6'
        self.options = {
            'page-size': 'A4',
            'margin-top': '0.2in',
            'margin-right': '0.2in',
            'margin-bottom': '0.2in',
            'margin-left': '0.2in'
        }

    def modify_html(self, html):
        """
        Modifies the HTML content.
        Adds Bootstrap table classes and styles for better formatting.

        Args:
            html (str): The original HTML content.

        Returns:
            str: The modified HTML content.
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Add Bootstrap table classes
        for table in soup.find_all('table'):
            table["class"] = self.classes

        # Add inline styles to table cells (optional, but can be helpful)
        for td in soup.find_all('td'):
            td["style"] = "font-size:10px;padding:2px;text-align:center;"

        for th in soup.find_all('th'):
            th["style"] = "font-size:10px;padding:2px;text-align:left;"

        # Add the main header to the body
        header = soup.new_tag('h1')
        header.string = 'WhatsApp Chat Analysis Report'
        soup.body.insert(0, header)

        return str(soup)

    def generate_pdf(self, content, wkhtmltopdf_path=None):
        """
        Generates the PDF report.

        Args:
            content (list): List of dictionaries, each representing a section.
            wkhtmltopdf_path (str): Optional path to wkhtmltopdf executable.
        """
        html_content = ""
        for section in content:
            if section["type"] == "html":
                html_content += section["data"]
            elif section["type"] == "table":
                html_content += section["data"].to_html(classes=self.classes, index=False)
            elif section["type"] == "image":
                width = section.get("width", 500)  # Default width 500
                height = section.get("height", 300)  # Default height 300
                # Check if the path is a URL (starts with http:// or https://)
                self.options.update({'enable-local-file-access': ''})
                if not section["data"].startswith("http://") and not section["data"].startswith("https://"):
                    # Convert relative path to absolute path
                    absolute_path = os.path.abspath(section["data"])
                    html_content += f'<img src="file:///{absolute_path}" width="{width}" height="{height}"><br>'
                else:
                    html_content += f'<img src="{section["data"]}" width="{width}" height="{height}"><br>'

        # Apply modify_html to the generated content
        modified_html = self.modify_html(html_content)

        # Insert the modified HTML into the template
        final_html = self.template.replace('%s', modified_html)

        html_filename = "tmp.html"
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(final_html)  # Write the final HTML

        if wkhtmltopdf_path:
            config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        else:
            config = pdfkit.configuration()
        try:
            pdfkit.from_file(
                html_filename, self.output_path, options=self.options, configuration=config
            )
            print(f"PDF report generated: {self.output_path}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
            traceback.print_exc()
        finally:
            os.remove(html_filename)  # Clean up temporary HTML file

# --- Helper Functions for Content Generation ---

def create_word_cloud_section(analyzer):
    analyzer.generate_wordcloud()
    wordcloud_filename = os.path.abspath("./data/wordcloud.png")  # Absolute path
    plt.savefig(wordcloud_filename)
    plt.close()
    return {
        "type": "image",
        "data": wordcloud_filename,
        "width": 500,
        "height": 300,
    }

def create_top_10_words_section(analyzer):
    word_frequencies = analyzer.calculate_word_frequency()
    top_10_words = word_frequencies.most_common(10)
    top_10_words_df = pd.DataFrame(top_10_words, columns=["Word", "Count"])
    return {
        "type": "html",
        "data": "<h3>Top 10 Most Frequent Words</h3>"
    }, {
        "type": "table",
        "data": top_10_words_df
    }

def create_message_length_section(analyzer):
    message_length_stats = analyzer.analyze_message_length()
    message_length_df = message_length_stats.reset_index()
    message_length_df.columns = ["Stat", "Value"]
    return {
        "type": "html",
        "data": "<h3>Message Length Distribution</h3>"
    }, {
        "type": "table",
        "data": message_length_df
    }

def create_media_count_section(analyzer):
    media_count = analyzer.analyze_media_count()
    return {
        "type": "html",
        "data": f"<h3>Total Media Messages: {media_count}</h3>"
    }

def create_top_5_emojis_section(analyzer):
    emoji_counts = analyzer.analyze_emoji_usage()
    top_5_emojis = emoji_counts.head(5)
    top_5_emojis_df = pd.DataFrame(top_5_emojis.reset_index())
    top_5_emojis_df.columns = ["Emoji", "Count"]
    return {
        "type": "html",
        "data": "<h3>Top 5 Most Frequent Emojis</h3>"
    }, {
        "type": "table",
        "data": top_5_emojis_df
    }

def create_plotly_chart_section(analyzer):
    fig = analyzer.create_plotly_fig(x="dayn", y="message", sortby=0)
    chart_filename = os.path.abspath("./data/plotly_chart.html")  # Absolute path
    # save the fig as html, instead of png
    fig.write_html(chart_filename)
    # open the html file and get its content
    with open(chart_filename, 'r', encoding='utf-8') as f:
        html_string = f.read()

    return {
        "type": "html",
        "data": html_string,
    }

def main():
    """
    Example usage of the WhatsApp Analyzer.
    """
    chat_file = "./data/whatsapp_chat.txt"
    wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'  # Update with your path
    html_template_path = 'templates/template.html'
    output_pdf_path = './data/whatsapp_report.pdf'

    try:
        parser = Parser(chat_file)
        df = parser.parse_chat_data()
        df = utils.df_basic_cleanup(df)
        analyzer = Analyzer(df)

        pdf_generator = PDFGenerator(html_template_path, output_pdf_path, wkhtmltopdf_path)

        # --- Content for the PDF (using helper functions) ---
        content = [
            {"type": "html", "data": "<h2>WhatsApp Chat Analysis</h2>"},
            create_word_cloud_section(analyzer),
            *create_top_10_words_section(analyzer),
            *create_message_length_section(analyzer),
            create_media_count_section(analyzer),
            *create_top_5_emojis_section(analyzer),
            create_plotly_chart_section(analyzer),
        ]

        # Generate the PDF
        pdf_generator.generate_pdf(content, wkhtmltopdf_path=wkhtmltopdf_path)

        # Clean up temporary image files
        for section in content:
            if section["type"] == "image":
                if os.path.exists(section["data"]):
                    os.remove(section["data"])

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()