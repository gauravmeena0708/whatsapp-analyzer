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
from whatsapp_analyzer.plot_utils import clean_message, analyze_emotion_over_time

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
        self.assertEqual(clean_message(msg), "Check   and more")

    def test_clean_message_empty(self):
        """Test empty and whitespace-only strings."""
        self.assertEqual(clean_message(""), "")
        self.assertEqual(clean_message("   "), "")

    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.pd')
    @patch('whatsapp_analyzer.plot_utils.TextBlob')
    def test_analyze_emotion_over_time_happy_path(self, mock_textblob, mock_pd, mock_plt, mock_plot_to_base64):
        """Test analyze_emotion_over_time with typical data."""
        mock_plot_to_base64.return_value = "fake_base64_image_data"

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.copy.return_value = mock_df

        # We need mock_df['message'] to return a MagicMock with an apply method
        # that just behaves normally and calls TextBlob mock
        mock_message_col = MagicMock()

        # When apply is called, we will just simulate getting sentiment series
        mock_sentiment_col = MagicMock()
        mock_emotion_col = MagicMock()

        mock_df.__getitem__.side_effect = lambda key: {
            'message': mock_message_col,
            'sentiment': mock_sentiment_col,
            'emotion': mock_emotion_col
        }.get(key, MagicMock())

        mock_message_col.apply.return_value = mock_sentiment_col
        mock_sentiment_col.apply.return_value = mock_emotion_col

        mock_daily_emotions = MagicMock()
        mock_daily_emotions.columns = ['joy', 'sadness']
        mock_daily_emotions.index = ['2023-01-01', '2023-01-02']

        mock_grouped = MagicMock()
        mock_df.groupby.return_value = mock_grouped

        # Mocking the grouping and unstacking chain
        mock_grouped.__getitem__.return_value.apply.return_value.unstack.return_value = mock_daily_emotions

        result = analyze_emotion_over_time(mock_df)

        self.assertEqual(result, "fake_base64_image_data")
        mock_df.set_index.assert_called_with('date', inplace=True)
        mock_plt.figure.assert_called()
        mock_plot_to_base64.assert_called_with(mock_plt)

    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.pd')
    def test_analyze_emotion_over_time_empty_df(self, mock_pd, mock_plt, mock_plot_to_base64):
        """Test analyze_emotion_over_time handles empty DataFrame."""
        mock_df = MagicMock()
        mock_df.empty = True
        mock_df.copy.return_value = mock_df

        result = analyze_emotion_over_time(mock_df)

        # Based on typical implementations, it should return an empty string or handle gracefully
        self.assertEqual(result, "")

    @patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @patch('whatsapp_analyzer.plot_utils.plt')
    @patch('whatsapp_analyzer.plot_utils.pd')
    @patch('whatsapp_analyzer.plot_utils.TextBlob')
    def test_analyze_emotion_over_time_with_username(self, mock_textblob, mock_pd, mock_plt, mock_plot_to_base64):
        """Test analyze_emotion_over_time filtering by username."""
        mock_plot_to_base64.return_value = "fake_base64_image_data_user"

        mock_df = MagicMock()
        mock_filtered_df = MagicMock()
        mock_filtered_df.empty = False

        # Setup the __getitem__ chain for boolean indexing: df[df['name'] == username]
        mock_name_col = MagicMock()
        mock_condition = MagicMock()

        def df_getitem_side_effect(key):
            if isinstance(key, MagicMock) and key is mock_condition:
                return mock_filtered_df
            elif key == 'name':
                return mock_name_col
            return MagicMock()

        mock_df.__getitem__.side_effect = df_getitem_side_effect
        mock_name_col.__eq__.return_value = mock_condition

        # The copy should return itself or a proper mock
        mock_filtered_df.copy.return_value = mock_filtered_df

        # For the groupby chain
        mock_daily_emotions = MagicMock()
        mock_daily_emotions.columns = ['joy']
        mock_daily_emotions.index = ['2023-01-01']
        mock_grouped = MagicMock()
        mock_filtered_df.groupby.return_value = mock_grouped
        mock_grouped.__getitem__.return_value.apply.return_value.unstack.return_value = mock_daily_emotions

        result = analyze_emotion_over_time(mock_df, username="Alice")

        self.assertEqual(result, "fake_base64_image_data_user")

if __name__ == '__main__':
    unittest.main()
