import unittest
import os
import tempfile
import shutil

from whatsapp_analyzer.analyzer import WhatsAppAnalyzer

# Base directory for all temporary test files (must be within project root)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestAnalyzerIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for output reports within the project tree
        self.test_out_dir = tempfile.mkdtemp(dir=_TESTS_DIR)

        # Create a temporary chat file within the project tree
        self.chat_fd, self.chat_file_path = tempfile.mkstemp(
            suffix=".txt", text=True, dir=_TESTS_DIR
        )

        # Write some sample chat data that the Parser can handle
        sample_chat = [
            "10/10/22, 10:00 AM - Alice: Hello everyone!\n",
            "10/10/22, 10:05 AM - Bob: Hi Alice.\n",
            "10/10/22, 10:10 AM - Alice: How are you doing? 😊\n",
            "10/10/22, 10:15 AM - Bob: I am doing great, thanks!\n",
            "10/10/22, 10:20 AM - System: Alice changed the group icon\n",
        ]

        with open(self.chat_file_path, "w", encoding="utf-8") as f:
            f.writelines(sample_chat)

    def tearDown(self):
        # Clean up the temporary chat file
        os.close(self.chat_fd)
        if os.path.exists(self.chat_file_path):
            os.remove(self.chat_file_path)

        # Clean up the temporary output directory
        shutil.rmtree(self.test_out_dir, ignore_errors=True)

    def test_analyzer_integration(self):
        """
        Integration test for WhatsAppAnalyzer.
        Initializes the analyzer with a real (temporary) file and output directory,
        runs generate_report, and verifies the generated HTML files.
        """
        analyzer = WhatsAppAnalyzer(self.chat_file_path, out_dir=self.test_out_dir)

        # Verify basic initialization and parsing
        self.assertEqual(analyzer.chat_file, self.chat_file_path)
        self.assertEqual(analyzer.out_dir, self.test_out_dir)
        self.assertIsNotNone(analyzer.df)
        self.assertFalse(analyzer.df.empty)

        # Generate reports for all users
        analyzer.generate_report()

        # Verify that reports were created for Alice and Bob, but not System
        expected_alice_report = os.path.join(self.test_out_dir, "Alice_report.html")
        expected_bob_report = os.path.join(self.test_out_dir, "Bob_report.html")
        unexpected_system_report = os.path.join(self.test_out_dir, "System_report.html")

        self.assertTrue(os.path.exists(expected_alice_report), "Alice's report should be generated.")
        self.assertTrue(os.path.exists(expected_bob_report), "Bob's report should be generated.")
        self.assertFalse(os.path.exists(unexpected_system_report), "System report should not be generated.")

        # Verify the content of the generated reports
        with open(expected_alice_report, "r", encoding="utf-8") as f:
            alice_html = f.read()
            self.assertIn("WhatsApp Chat Analysis - Alice", alice_html)

        with open(expected_bob_report, "r", encoding="utf-8") as f:
            bob_html = f.read()
            self.assertIn("WhatsApp Chat Analysis - Bob", bob_html)

    def test_generate_report_specific_users(self):
        """
        Test generating a report for a specific user, and test filename sanitization.
        """
        # Append a message from a user with special characters
        with open(self.chat_file_path, "a", encoding="utf-8") as f:
            f.write("10/10/22, 10:25 AM - Charlie/Chaplin*<>: Hello there!\n")

        analyzer = WhatsAppAnalyzer(self.chat_file_path, out_dir=self.test_out_dir)

        # Generate report only for the specific user
        dirty_name = "Charlie/Chaplin*<>"
        analyzer.generate_report(users=[dirty_name])

        # The re is r'[^a-zA-Z0-9_\- ]' -> replaced with '_'
        # "Charlie/Chaplin*<>" has 4 invalid characters: /, *, <, >
        expected_safe_name = "Charlie_Chaplin___"
        expected_report = os.path.join(self.test_out_dir, f"{expected_safe_name}_report.html")

        self.assertTrue(
            os.path.exists(expected_report),
            f"Report for sanitized name {expected_safe_name} should be generated.",
        )

        # Other users should not have reports generated
        self.assertFalse(os.path.exists(os.path.join(self.test_out_dir, "Alice_report.html")))
        self.assertFalse(os.path.exists(os.path.join(self.test_out_dir, "Bob_report.html")))


class TestWhatsAppAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_base_dir = tempfile.mkdtemp(dir=_TESTS_DIR)
        self.chat_path = os.path.join(self.test_base_dir, "chat.txt")
        self.output_dir = os.path.join(self.test_base_dir, "reports")

    def tearDown(self):
        shutil.rmtree(self.test_base_dir, ignore_errors=True)

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
