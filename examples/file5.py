# usage.py

import os
import traceback
from whatsapp_analyzer.parser import Parser
from whatsapp_analyzer import utils
from whatsapp_analyzer.analyzer import Analyzer
from whatsapp_analyzer.pdf_generator import PDFGenerator  # Import PDFGenerator from the package
from whatsapp_analyzer.report_sections import (
    create_word_cloud_section,
    create_top_10_words_section,
    create_message_length_section,
    create_media_count_section,
    create_top_5_emojis_section,
    create_seaborn_chart_section
)
from bs4 import BeautifulSoup
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
