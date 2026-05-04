import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import sys
from bs4 import BeautifulSoup

class TestPDFGenerator(unittest.TestCase):
    def setUp(self):
        # Patch sys.modules to mock heavy dependencies only during test execution
        self.mock_modules = {
            'pdfkit': MagicMock(),
            'nltk': MagicMock(),
            'nltk.corpus': MagicMock(),
            'nltk.sentiment.vader': MagicMock(),
            'textblob': MagicMock(),
            'pandas': MagicMock(),
            'wordcloud': MagicMock(),
            'networkx': MagicMock(),
            'sklearn': MagicMock(),
            'sklearn.feature_extraction.text': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'matplotlib.colors': MagicMock(),
            'matplotlib.font_manager': MagicMock()
        }
        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()

        # Now import PDFGenerator safely
        from whatsapp_analyzer.pdf_generator import PDFGenerator
        self.PDFGenerator = PDFGenerator

        self.template_content = "<html><body>%s</body></html>"
        self.temp_template_fd, self.temp_template_path = tempfile.mkstemp(suffix=".html", text=True)
        with os.fdopen(self.temp_template_fd, 'w', encoding='utf-8') as f:
            f.write(self.template_content)

        # Use safe temporary file creation instead of deprecated mktemp
        self.temp_output_fd, self.temp_output_path = tempfile.mkstemp(suffix=".pdf")
        os.close(self.temp_output_fd) # Close it, pdfkit will write to it

        self.wkhtmltopdf_path = "/usr/bin/wkhtmltopdf"

    def tearDown(self):
        self.patcher.stop()

        if os.path.exists(self.temp_template_path):
            os.remove(self.temp_template_path)
        if os.path.exists(self.temp_output_path):
            os.remove(self.temp_output_path)
        if os.path.exists("tmp.html"):
            os.remove("tmp.html")

    def test_init(self):
        generator = self.PDFGenerator(self.temp_template_path, self.temp_output_path, self.wkhtmltopdf_path)
        self.assertEqual(generator.template, self.template_content)
        self.assertEqual(generator.output_path, self.temp_output_path)
        self.assertEqual(generator.wkhtmltopdf_path, self.wkhtmltopdf_path)
        self.assertEqual(generator.classes, 'table table-sm table-bordered border-primary d-print-table fs-6')

    def test_modify_html_adds_body(self):
        generator = self.PDFGenerator(self.temp_template_path, self.temp_output_path, self.wkhtmltopdf_path)
        html = "<h1>Hello</h1>"
        modified_html = generator.modify_html(html)
        self.assertIn("<body>", modified_html)
        self.assertIn("<h1>Hello</h1>", modified_html)

    def test_modify_html_modifies_tables(self):
        generator = self.PDFGenerator(self.temp_template_path, self.temp_output_path, self.wkhtmltopdf_path)
        html = "<body><table><tr><th>Header</th></tr><tr><td>Data</td></tr></table></body>"
        modified_html = generator.modify_html(html)

        soup = BeautifulSoup(modified_html, 'html.parser')
        table = soup.find('table')
        self.assertIsNotNone(table)
        self.assertEqual(" ".join(table.get('class', [])), generator.classes)

        td = soup.find('td')
        self.assertIsNotNone(td)
        self.assertEqual(td.get('style'), "font-size:10px;padding:2px;text-align:center;")

        th = soup.find('th')
        self.assertIsNotNone(th)
        self.assertEqual(th.get('style'), "font-size:10px;padding:2px;text-align:left;")

    @patch('whatsapp_analyzer.pdf_generator.pdfkit')
    def test_generate_pdf(self, mock_pdfkit):
        generator = self.PDFGenerator(self.temp_template_path, self.temp_output_path, self.wkhtmltopdf_path)

        mock_df = MagicMock()
        mock_df.to_html.return_value = "<table><tr><td>Mock Table Data</td></tr></table>"

        content = [
            {"type": "html", "data": "<h2>Report</h2>"},
            {"type": "table", "data": mock_df},
            {"type": "image", "data": "https://example.com/image.png", "width": 100, "height": 100},
            {"type": "image", "data": "local_image.png"}
        ]

        generator.generate_pdf(content)

        mock_pdfkit.configuration.assert_called_once_with(wkhtmltopdf=self.wkhtmltopdf_path)
        mock_pdfkit.from_file.assert_called_once()
        args, kwargs = mock_pdfkit.from_file.call_args
        self.assertEqual(args[0], "tmp.html")
        self.assertEqual(args[1], self.temp_output_path)

        self.assertTrue(os.path.exists("tmp.html"))
        with open("tmp.html", "r", encoding="utf-8") as f:
            tmp_content = f.read()

        self.assertIn("<h2>Report</h2>", tmp_content)
        self.assertIn("Mock Table Data", tmp_content)
        self.assertIn("https://example.com/image.png", tmp_content)

        absolute_path = os.path.abspath("local_image.png")
        self.assertIn(f"file:///{absolute_path}", tmp_content)

    @patch('whatsapp_analyzer.pdf_generator.pdfkit')
    def test_generate_pdf_handles_exception(self, mock_pdfkit):
        generator = self.PDFGenerator(self.temp_template_path, self.temp_output_path, self.wkhtmltopdf_path)
        mock_pdfkit.from_file.side_effect = Exception("Mocked exception")

        content = [{"type": "html", "data": "<h2>Report</h2>"}]

        # Test that the exception is caught and printed (we can check standard output but testing that it doesn't crash is good)
        try:
            generator.generate_pdf(content)
        except Exception as e:
            self.fail(f"generate_pdf raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
