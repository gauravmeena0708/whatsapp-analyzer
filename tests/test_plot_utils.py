# tests/test_plot_utils.py

import unittest
from unittest.mock import MagicMock, patch
import sys

# Mocking dependencies to allow importing plot_utils
sys.modules['pandas'] = MagicMock()
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
from whatsapp_analyzer.plot_utils import clean_message, analyze_sentiment_over_time

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

    @patch('whatsapp_analyzer.plot_utils.TextBlob')
    @patch('whatsapp_analyzer.plot_utils.pd')
    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling')
    def test_analyze_sentiment_over_time(self, mock_apply_styling, mock_plot_to_base64, mock_plt, mock_pd, mock_textblob):
        """Test analyze_sentiment_over_time function without a username."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()
        mock_df.copy.return_value = mock_df_filtered

        mock_message_col = MagicMock()
        mock_sentiment_col = MagicMock()

        def getitem_side_effect(key):
            if key == 'message': return mock_message_col
            if key == 'sentiment': return mock_sentiment_col
            if key == 'date': return MagicMock()
            return MagicMock()

        mock_df_filtered.__getitem__.side_effect = getitem_side_effect

        mock_df_filtered.empty = False

        mock_resampler = MagicMock()
        mock_mean_result = MagicMock()
        mock_mean_result.index = [1, 2]
        mock_mean_result.values = [0.1, 0.2]
        mock_resampler.mean.return_value = mock_mean_result
        mock_sentiment_col.resample.return_value = mock_resampler

        mock_plot_to_base64.return_value = "base64_encoded_string"

        result = analyze_sentiment_over_time(mock_df)

        self.assertEqual(result, "base64_encoded_string")
        mock_df.copy.assert_called_once()
        mock_df_filtered.set_index.assert_called_once_with('date', inplace=True)
        mock_sentiment_col.resample.assert_called_once_with('W')
        mock_resampler.mean.assert_called_once()
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_apply_styling.assert_called_once()
        mock_plot_to_base64.assert_called_once_with(mock_plt)

    def test_analyze_sentiment_over_time_empty(self):
        """Test analyze_sentiment_over_time with empty dataframe."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()
        mock_df.copy.return_value = mock_df_filtered
        mock_df_filtered.empty = True

        result = analyze_sentiment_over_time(mock_df)
        self.assertEqual(result, "")

    @patch('whatsapp_analyzer.plot_utils.TextBlob')
    @patch('whatsapp_analyzer.plot_utils.pd')
    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.apply_consistent_plot_styling')
    def test_analyze_sentiment_over_time_with_username(self, mock_apply_styling, mock_plot_to_base64, mock_plt, mock_pd, mock_textblob):
        """Test analyze_sentiment_over_time function with a username."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()

        mock_filter_result = MagicMock()
        mock_filter_result.copy.return_value = mock_df_filtered
        mock_df.__getitem__.return_value = mock_filter_result

        mock_message_col = MagicMock()
        mock_sentiment_col = MagicMock()

        def getitem_side_effect(key):
            if key == 'message': return mock_message_col
            if key == 'sentiment': return mock_sentiment_col
            if key == 'date': return MagicMock()
            return MagicMock()

        mock_df_filtered.__getitem__.side_effect = getitem_side_effect

        mock_df_filtered.empty = False

        mock_resampler = MagicMock()
        mock_mean_result = MagicMock()
        mock_mean_result.index = [1, 2]
        mock_mean_result.values = [0.1, 0.2]
        mock_resampler.mean.return_value = mock_mean_result
        mock_sentiment_col.resample.return_value = mock_resampler

        mock_plot_to_base64.return_value = "base64_encoded_string"

        result = analyze_sentiment_over_time(mock_df, username="Alice")

        self.assertEqual(result, "base64_encoded_string")
        mock_filter_result.copy.assert_called_once()
        mock_df_filtered.set_index.assert_called_once_with('date', inplace=True)
        mock_sentiment_col.resample.assert_called_once_with('W')
        mock_resampler.mean.assert_called_once()
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_apply_styling.assert_called_once()
        mock_plot_to_base64.assert_called_once_with(mock_plt)


if __name__ == '__main__':
    unittest.main()
