# whatsapp_analyzer/pdf_generator.py

import os
import pdfkit
from pathlib import Path
from bs4 import BeautifulSoup
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
            'margin-left': '0.2in',
            'enable-local-file-access': ''
        }

    def modify_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')

        if soup.body is None:
            body_tag = soup.new_tag('body')
            soup.append(body_tag)
        else:
            body_tag = soup.body

        for table in soup.find_all('table'):
            table["class"] = self.classes

        for td in soup.find_all('td'):
            td["style"] = "font-size:10px;padding:2px;text-align:center;"

        for th in soup.find_all('th'):
            th["style"] = "font-size:10px;padding:2px;text-align:left;"

        return str(soup)

    def generate_pdf(self, content, wkhtmltopdf_path=None):
        html_content = ""
        for section in content:
            if section["type"] == "html":
                html_content += section["data"]
            elif section["type"] == "table":
                html_content += section["data"].to_html(classes=self.classes, index=False)
            elif section["type"] == "image":
                width = section.get("width", 500)
                height = section.get("height", 300)
                self.options.update({'enable-local-file-access': ''})

                if not section["data"].startswith("http://") and not section["data"].startswith("https://"):
                    absolute_path = os.path.abspath(section["data"])
                    html_content += f'<img src="file:///{absolute_path}" width="{width}" height="{height}"><br>'
                else:
                    html_content += f'<img src="{section["data"]}" width="{width}" height="{height}"><br>'

        modified_html = self.modify_html(html_content)
        final_html = self.template.replace('%s', modified_html)

        html_filename = "tmp.html"
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(final_html)

        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path or self.wkhtmltopdf_path)
        try:
            pdfkit.from_file(html_filename, self.output_path, options=self.options, configuration=config)
            print(f"PDF report generated: {self.output_path}")
        except Exception as e:
            print(f"Error generating PDF: {e}")
            traceback.print_exc()

        #os.remove(html_filename)  # Clean up temporary HTML file
