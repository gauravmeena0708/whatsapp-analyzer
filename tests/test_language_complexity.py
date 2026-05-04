import unittest
import pandas as pd
from whatsapp_analyzer.plot_utils import analyze_language_complexity
import base64

class TestLanguageComplexity(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame that reflects actual message data
        self.data = {
            'date_time': pd.to_datetime([
                '2023-01-01 10:00:00',
                '2023-01-01 10:05:00',
                '2023-01-01 10:10:00',
                '2023-01-02 11:00:00',
                '2023-01-02 11:05:00'
            ]),
            'name': ['Alice', 'Bob', 'Alice', 'Bob', 'Charlie'],
            'message': [
                'Hello! How are you doing today?',
                'I am good, thanks. How about you?',
                'I am doing great. Just working on some code.',
                'Awesome. Let me know if you need any help.',
                'Hey guys, what is going on here? 😊'
            ]
        }
        self.df = pd.DataFrame(self.data)

    def test_analyze_language_complexity_empty_df(self):
        """Test with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['date_time', 'name', 'message'])
        result = analyze_language_complexity(empty_df)
        self.assertEqual(result, "")

    def test_analyze_language_complexity_no_username(self):
        """Test without filtering by username."""
        result = analyze_language_complexity(self.df)
        # Should return a base64 encoded string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

        # Verify it's valid base64
        try:
            base64.b64decode(result)
        except Exception:
            self.fail("Result is not a valid base64 string")

    def test_analyze_language_complexity_with_username(self):
        """Test filtering by a specific username."""
        result = analyze_language_complexity(self.df, username='Alice')
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

        # Alice's messages should be successfully processed
        # A different base64 plot would be generated, though we just check format
        try:
            base64.b64decode(result)
        except Exception:
            self.fail("Result is not a valid base64 string")

    def test_analyze_language_complexity_only_emojis(self):
        """Test behavior when a user's messages only contain emojis or short words."""
        emoji_df = pd.DataFrame({
            'date_time': pd.to_datetime(['2023-01-01 10:00:00']),
            'name': ['Dave'],
            'message': ['😊 😂 ❤️ a i']
        })
        result = analyze_language_complexity(emoji_df)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()
