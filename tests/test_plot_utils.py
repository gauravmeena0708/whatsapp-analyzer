# tests/test_plot_utils.py

import unittest
from unittest.mock import MagicMock
import sys
import pandas as pd

# Mocking other heavy dependencies we don't need for this specific test
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
from whatsapp_analyzer.plot_utils import clean_message, plot_most_active_hours
from unittest.mock import patch

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


class TestPlotMostActiveHours(unittest.TestCase):

    def setUp(self):
        # Create a small DataFrame for testing
        data = {
            'name': ['Alice', 'Bob', 'Alice', 'Bob', 'Alice'],
            'hour': [8, 9, 8, 10, 11]
        }
        self.df = pd.DataFrame(data)

    def test_plot_most_active_hours_empty_df(self):
        """Test when the resulting DataFrame is empty."""
        empty_df = pd.DataFrame(columns=['name', 'hour'])
        result = plot_most_active_hours(empty_df)
        self.assertEqual(result, "")

    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling')
    def test_plot_most_active_hours_no_username(self, mock_apply_styling, mock_plot_to_base64, mock_plt):
        """Test plot_most_active_hours without a username."""
        mock_plot_to_base64.return_value = "base64_string"

        result = plot_most_active_hours(self.df)

        # In self.df, hours are: 8 (2 times), 9 (1 time), 10 (1 time), 11 (1 time)
        # value_counts().sort_index() would yield:
        # 8: 2, 9: 1, 10: 1, 11: 1

        mock_plt.figure.assert_called_once_with(figsize=(12, 6), constrained_layout=True)

        # Get the arguments passed to plt.bar
        bar_args, bar_kwargs = mock_plt.bar.call_args

        # Verify the x and y values for the bar plot
        # Using list() to convert pandas Index/Array for comparison
        self.assertEqual(list(bar_args[0]), [8, 9, 10, 11])
        self.assertEqual(list(bar_args[1]), [2, 1, 1, 1])
        self.assertEqual(bar_kwargs, {'color': 'skyblue'})

        mock_apply_styling.assert_called_once_with(mock_plt, 'Most Active Hours ', 'Hour of the Day', 'Number of Messages')
        mock_plot_to_base64.assert_called_once_with(mock_plt)

        self.assertEqual(result, "base64_string")

    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling')
    def test_plot_most_active_hours_with_username(self, mock_apply_styling, mock_plot_to_base64, mock_plt):
        """Test plot_most_active_hours with a specific username."""
        mock_plot_to_base64.return_value = "base64_string_user"

        result = plot_most_active_hours(self.df, username="Alice")

        # For Alice, hours are: 8 (2 times), 11 (1 time)
        mock_plt.figure.assert_called_once_with(figsize=(12, 6), constrained_layout=True)

        bar_args, bar_kwargs = mock_plt.bar.call_args
        self.assertEqual(list(bar_args[0]), [8, 11])
        self.assertEqual(list(bar_args[1]), [2, 1])
        self.assertEqual(bar_kwargs, {'color': 'skyblue'})

        mock_apply_styling.assert_called_once_with(mock_plt, 'Most Active Hours for Alice', 'Hour of the Day', 'Number of Messages')
        mock_plot_to_base64.assert_called_once_with(mock_plt)

        self.assertEqual(result, "base64_string_user")

    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.plt')
    def test_plot_most_active_hours_username_not_found(self, mock_plt, mock_plot_to_base64):
        """Test with a username that doesn't exist in the DataFrame."""
        result = plot_most_active_hours(self.df, username="Charlie")

        # Should return an empty string because user_df will be empty
        self.assertEqual(result, "")
        mock_plt.figure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
