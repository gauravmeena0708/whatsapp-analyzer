# tests/test_plot_utils.py

import unittest
from unittest.mock import MagicMock
import sys

# Mocking dependencies to allow importing plot_utils
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.font_manager'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['textblob'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.feature_extraction'] = MagicMock()
sys.modules['sklearn.feature_extraction.text'] = MagicMock()
sys.modules['wordcloud'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.corpus'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['emoji'] = MagicMock()

# Now we can import the function to test
from whatsapp_analyzer.plot_utils import clean_message, plot_emoji_usage
import pandas as pd
import emoji

class TestPlotUtils(unittest.TestCase):

    def test_clean_message_basic(self):
        """Test cleaning a basic message."""
        self.assertEqual(clean_message("Hello world"), "Hello world")

    def test_clean_message_strip(self):
        """Test stripping leading/trailing whitespace."""
        self.assertEqual(clean_message("  Hello world  "), "Hello world")

    def test_clean_message_url_http(self):
        """Test removing http URLs."""
        self.assertEqual(clean_message("Check this http://example.com"), "Check this")
        self.assertEqual(clean_message("http://example.com is cool"), "is cool")

    def test_clean_message_url_https(self):
        """Test removing https URLs."""
        self.assertEqual(clean_message("Secure link https://test.org/path?q=1"), "Secure link")

    def test_clean_message_media_omitted(self):
        """Test removing 'media omitted' phrases."""
        self.assertEqual(clean_message("This was media omitted"), "This was")
        self.assertEqual(clean_message("MEDIA OMITTED and some text"), "and some text")
        self.assertEqual(clean_message("Mixed Case Media Omitted"), "Mixed Case")

    def test_clean_message_bracketed_media_omitted(self):
        """Test removing '<media omitted>' phrases."""
        self.assertEqual(clean_message("See <media omitted> here"), "See  here")
        self.assertEqual(clean_message("<MEDIA OMITTED>"), "")

    def test_clean_message_combination(self):
        """Test combination of URL and media omitted."""
        msg = "Check http://link.com <media omitted> and more"
        # Current implementation:
        # 1. Remove URLs -> "Check  <media omitted> and more"
        # 2. Remove media omitted -> "Check   and more"
        # 3. Strip -> "Check   and more" (note: internal spaces are preserved)
        self.assertEqual(clean_message(msg), "Check   and more")

    def test_clean_message_empty(self):
        """Test empty and whitespace-only strings."""
        self.assertEqual(clean_message(""), "")
        self.assertEqual(clean_message("   "), "")

    # Tests for plot_emoji_usage

    def test_plot_emoji_usage_basic(self):
        """Test emoji plot generation with emojis."""
        # Need to mock emoji.EMOJI_DATA because emoji module is mocked
        import emoji
        original_emoji_data = getattr(emoji, 'EMOJI_DATA', {})
        # ❤️ is actually two characters: '❤' and '️' in Python strings depending on encoding.
        # Let's use simple single-character emojis for testing.
        emoji.EMOJI_DATA = {'😀': {}, '😂': {}, '🚀': {}}

        try:
            mock_df = pd.DataFrame({'message': ['Hello 😀', 'Hi 😂', '🚀 🚀 🚀']})

            with unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64') as mock_plot_to_base64, \
                 unittest.mock.patch('whatsapp_analyzer.plot_utils.plt') as mock_plt, \
                 unittest.mock.patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling') as mock_styling:

                mock_plot_to_base64.return_value = "base64_string"

                result = plot_emoji_usage(mock_df)

                self.assertEqual(result, "base64_string")
                # Verify bar was called with correct top emojis: 🚀 (3), 😀 (1), 😂 (1)
                # Counter().most_common(5) order for ties depends on insertion order.
                mock_plt.bar.assert_called_once()
                args, kwargs = mock_plt.bar.call_args
                self.assertEqual(args[0][0], '🚀') # most common
                self.assertEqual(args[1][0], 3)
                self.assertIn('😀', args[0])
                self.assertIn('😂', args[0])
        finally:
            emoji.EMOJI_DATA = original_emoji_data

    def test_plot_emoji_usage_no_emojis(self):
        """Test emoji plot generation when no emojis are found."""
        mock_df = pd.DataFrame({'message': ['Hello', 'Hi', 'Bye']})

        with unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64') as mock_plot_to_base64, \
             unittest.mock.patch('whatsapp_analyzer.plot_utils.plt') as mock_plt, \
             unittest.mock.patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling') as mock_styling:

            mock_plot_to_base64.return_value = "base64_string_no_emoji"

            result = plot_emoji_usage(mock_df)

            self.assertEqual(result, "base64_string_no_emoji")
            # Verify text is displayed saying "No emojis found."
            mock_plt.text.assert_called_once_with(0.5, 0.5, "No emojis found.", ha='center', va='center', fontsize=12)
            mock_plt.bar.assert_not_called()

    def test_plot_emoji_usage_empty_df(self):
        """Test emoji plot generation with an empty dataframe."""
        mock_df = pd.DataFrame({'message': []})

        with unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64') as mock_plot_to_base64, \
             unittest.mock.patch('whatsapp_analyzer.plot_utils.plt') as mock_plt, \
             unittest.mock.patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling') as mock_styling:

            result = plot_emoji_usage(mock_df)

            # The function returns "" when the filtered dataframe is empty
            self.assertEqual(result, "")
            mock_plt.text.assert_not_called()
            mock_plt.bar.assert_not_called()

    def test_plot_emoji_usage_with_username(self):
        """Test emoji plot filtering by username."""
        mock_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Alice'], 'message': ['Hello 😀', 'Hi', 'Bye 😂']})

        with unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64') as mock_plot_to_base64, \
             unittest.mock.patch('whatsapp_analyzer.plot_utils.plt') as mock_plt, \
             unittest.mock.patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling') as mock_styling:

            mock_plot_to_base64.return_value = "base64_string_user"

            result = plot_emoji_usage(mock_df, username="Alice")

            self.assertEqual(result, "base64_string_user")
            mock_styling.assert_called_once()
            args, kwargs = mock_styling.call_args
            self.assertEqual(args[1], 'Emoji Usage for Alice')

if __name__ == '__main__':
    unittest.main()
