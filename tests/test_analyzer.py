import os
import tempfile
import unittest

from whatsapp_analyzer.analyzer import WhatsAppAnalyzer


class TestWhatsAppAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chat_path = os.path.join(self.temp_dir.name, "chat.txt")
        self.output_dir = os.path.join(self.temp_dir.name, "reports")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_chat(self, lines):
        with open(self.chat_path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    def test_generate_report_creates_sanitized_html_files(self):
        self._write_chat(
            [
                "20/03/2023, 10:00 - Alice/Dev: Hello there!",
                "20/03/2023, 10:01 - Bob: Hi Alice.",
                "20/03/2023, 10:02 - Alice/Dev: Great to see you 😊",
                "20/03/2023, 10:03 - Bob: This is good news!",
            ]
        )

        analyzer = WhatsAppAnalyzer(chat_file=self.chat_path, out_dir=self.output_dir)
        analyzer.generate_report(users=["Alice/Dev"])

        report_path = os.path.join(self.output_dir, "Alice_Dev_report.html")
        self.assertTrue(os.path.exists(report_path))

        with open(report_path, "r", encoding="utf-8") as handle:
            html = handle.read()

        self.assertIn("Alice/Dev", html)
        self.assertIn("data:image/png;base64", html)


if __name__ == "__main__":
    unittest.main()
