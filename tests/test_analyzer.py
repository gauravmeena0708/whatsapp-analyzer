import unittest
import os
import tempfile
import shutil

# No mocking of sys.modules for external dependencies.
# The code should be imported cleanly at the top of the file.
from whatsapp_analyzer.analyzer import WhatsAppAnalyzer

class TestAnalyzerIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for output reports
        self.test_out_dir = tempfile.mkdtemp()

        # Create a temporary chat file
        self.chat_fd, self.chat_file_path = tempfile.mkstemp(suffix=".txt", text=True)

        # Write some sample chat data that the Parser can handle
        sample_chat = [
            "10/10/22, 10:00 AM - Alice: Hello everyone!\n",
            "10/10/22, 10:05 AM - Bob: Hi Alice.\n",
            "10/10/22, 10:10 AM - Alice: How are you doing? 😊\n",
            "10/10/22, 10:15 AM - Bob: I am doing great, thanks!\n",
            "10/10/22, 10:20 AM - System: Alice changed the group icon\n"
        ]

        with open(self.chat_file_path, "w", encoding="utf-8") as f:
            f.writelines(sample_chat)

    def tearDown(self):
        # Clean up the temporary chat file
        os.close(self.chat_fd)
        if os.path.exists(self.chat_file_path):
            os.remove(self.chat_file_path)

        # Clean up the temporary output directory
        shutil.rmtree(self.test_out_dir)

    def test_analyzer_integration(self):
        """
        Integration test for WhatsAppAnalyzer.
        Initializes the analyzer with a real (temporary) file and output directory,
        runs generate_report, and verifies the generated HTML files.
        """
        # Initialize the analyzer
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
        # We'll append a message from a user with special characters
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

        self.assertTrue(os.path.exists(expected_report), f"Report for sanitized name {expected_safe_name} should be generated.")

        # Other users should not have reports generated
        self.assertFalse(os.path.exists(os.path.join(self.test_out_dir, "Alice_report.html")))
        self.assertFalse(os.path.exists(os.path.join(self.test_out_dir, "Bob_report.html")))

if __name__ == '__main__':
    unittest.main()
