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
sys.modules['numpy'] = MagicMock()

# Now we can import the function to test
from whatsapp_analyzer.plot_utils import clean_message, generate_wordcloud, analyze_emotion_over_time

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


    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.WordCloud')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plt')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils._filter_by_user')
    def test_generate_wordcloud_basic(self, mock_filter_by_user, mock_plt, mock_wordcloud, mock_plot_to_base64):
        """Test generating a word cloud for all users."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()
        mock_df_filtered.empty = False
        mock_filter_by_user.return_value = mock_df_filtered

        mock_clean_message = MagicMock()
        mock_clean_message.__iter__.return_value = iter(['hello world', 'good morning'])

        def getitem_side_effect(key):
            if key == 'clean_message':
                return mock_clean_message
            return MagicMock()

        mock_df_filtered.__getitem__.side_effect = getitem_side_effect
        mock_plot_to_base64.return_value = "base64_encoded_string"

        result = generate_wordcloud(mock_df)

        self.assertEqual(result, "base64_encoded_string")
        mock_wordcloud.assert_called()
        mock_plot_to_base64.assert_called_once_with(mock_plt)

    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.WordCloud')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plt')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils._filter_by_user')
    def test_generate_wordcloud_with_username(self, mock_filter_by_user, mock_plt, mock_wordcloud, mock_plot_to_base64):
        """Test generating a word cloud for a specific user."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()
        mock_df_filtered.empty = False
        mock_filter_by_user.return_value = mock_df_filtered

        mock_clean_message = MagicMock()
        mock_clean_message.__iter__.return_value = iter(['specific user message'])

        def getitem_side_effect(key):
            if key == 'clean_message':
                return mock_clean_message
            return MagicMock()

        mock_df_filtered.__getitem__.side_effect = getitem_side_effect
        mock_plot_to_base64.return_value = "base64_specific_user"

        result = generate_wordcloud(mock_df, username="Alice")

        self.assertEqual(result, "base64_specific_user")
        mock_wordcloud.assert_called()
        mock_plot_to_base64.assert_called_once_with(mock_plt)

    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.WordCloud')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plt')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils._filter_by_user')
    def test_generate_wordcloud_empty_text(self, mock_filter_by_user, mock_plt, mock_wordcloud, mock_plot_to_base64):
        """Test generating a word cloud when text is empty."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()
        mock_df_filtered.empty = False
        mock_filter_by_user.return_value = mock_df_filtered

        # Empty iterator to simulate no text
        mock_clean_message = MagicMock()
        mock_clean_message.__iter__.return_value = iter([])

        def getitem_side_effect(key):
            if key == 'clean_message':
                return mock_clean_message
            return MagicMock()

        mock_df_filtered.__getitem__.side_effect = getitem_side_effect
        mock_plot_to_base64.return_value = "base64_empty"

        result = generate_wordcloud(mock_df)

        self.assertEqual(result, "base64_empty")
        mock_wordcloud.assert_not_called()
        mock_plt.text.assert_called_once_with(0.5, 0.5, "No words to display in word cloud.", ha='center', va='center', fontsize=12)
        mock_plot_to_base64.assert_called_once_with(mock_plt)

    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plot_to_base64')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.WordCloud')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils.plt')
    @unittest.mock.patch('whatsapp_analyzer.plot_utils._filter_by_user')
    def test_generate_wordcloud_value_error(self, mock_filter_by_user, mock_plt, mock_wordcloud, mock_plot_to_base64):
        """Test generating a word cloud when WordCloud throws ValueError."""
        mock_df = MagicMock()
        mock_df_filtered = MagicMock()
        mock_df_filtered.empty = False
        mock_filter_by_user.return_value = mock_df_filtered

        mock_clean_message = MagicMock()
        mock_clean_message.__iter__.return_value = iter(['test message'])

        def getitem_side_effect(key):
            if key == 'clean_message':
                return mock_clean_message
            return MagicMock()

        mock_df_filtered.__getitem__.side_effect = getitem_side_effect
        mock_plot_to_base64.return_value = "base64_error"

        # Make WordCloud raise ValueError
        mock_wordcloud.return_value.generate.side_effect = ValueError("Test Error")

        result = generate_wordcloud(mock_df)

        self.assertEqual(result, "base64_error")
        mock_wordcloud.assert_called()
        mock_plt.text.assert_called_once_with(0.5, 0.5, "Could not generate word cloud:\nTest Error", ha='center', va='center', fontsize=12, color='red')
        mock_plot_to_base64.assert_called_once_with(mock_plt)




if __name__ == '__main__':
    unittest.main()

    @patch('whatsapp_analyzer.plot_utils.render_chartjs')
    @patch('whatsapp_analyzer.plot_utils._filter_by_user')
    @patch('whatsapp_analyzer.plot_utils.pd')
    @patch('whatsapp_analyzer.plot_utils._polarity_subjectivity')
    def test_analyze_emotion_over_time_happy_path(self, mock_polarity, mock_pd, mock_filter_by_user, mock_render_chartjs):
        """Test analyze_emotion_over_time with typical data."""
        mock_render_chartjs.return_value = "rendered_chartjs_html"

        mock_df = MagicMock()
        mock_filtered_df = MagicMock()
        mock_filtered_df.empty = False
        mock_filter_by_user.return_value = mock_filtered_df

        mock_filtered_df.columns = ['message', 'date']

        mock_sentiment_col = MagicMock()
        mock_emotion_col = MagicMock()

        mock_filtered_df.__getitem__.side_effect = lambda key: {
            'message': MagicMock(),
            'sentiment': mock_sentiment_col,
            'emotion': mock_emotion_col
        }.get(key, MagicMock())

        mock_daily_emotions = MagicMock()
        mock_daily_emotions.columns = ['joy', 'sadness']
        mock_daily_emotions.index = [MagicMock(), MagicMock()]

        mock_grouped = MagicMock()
        mock_filtered_df.groupby.return_value = mock_grouped

        mock_grouped.__getitem__.return_value.apply.return_value.unstack.return_value.resample.return_value.sum.return_value = mock_daily_emotions

        result = analyze_emotion_over_time(mock_df)

        self.assertEqual(result, "rendered_chartjs_html")
        mock_filtered_df.set_index.assert_called_with('date', inplace=True)
        mock_render_chartjs.assert_called()

    @patch('whatsapp_analyzer.plot_utils._filter_by_user')
    def test_analyze_emotion_over_time_empty_df(self, mock_filter_by_user):
        """Test analyze_emotion_over_time handles empty DataFrame."""
        mock_df = MagicMock()
        mock_filtered_df = MagicMock()
        mock_filtered_df.empty = True
        mock_filter_by_user.return_value = mock_filtered_df

        result = analyze_emotion_over_time(mock_df)

        self.assertEqual(result, "<p class='text-center text-muted'>No messages available for emotion trend analysis.</p>")

    @patch('whatsapp_analyzer.plot_utils.render_chartjs')
    @patch('whatsapp_analyzer.plot_utils._filter_by_user')
    @patch('whatsapp_analyzer.plot_utils.pd')
    @patch('whatsapp_analyzer.plot_utils._polarity_subjectivity')
    def test_analyze_emotion_over_time_with_username(self, mock_polarity, mock_pd, mock_filter_by_user, mock_render_chartjs):
        """Test analyze_emotion_over_time filtering by username."""
        mock_render_chartjs.return_value = "rendered_chartjs_html_user"

        mock_df = MagicMock()
        mock_filtered_df = MagicMock()
        mock_filtered_df.empty = False
        mock_filter_by_user.return_value = mock_filtered_df

        mock_filtered_df.columns = ['message', 'date']

        mock_sentiment_col = MagicMock()
        mock_emotion_col = MagicMock()

        mock_filtered_df.__getitem__.side_effect = lambda key: {
            'message': MagicMock(),
            'sentiment': mock_sentiment_col,
            'emotion': mock_emotion_col
        }.get(key, MagicMock())

        mock_daily_emotions = MagicMock()
        mock_daily_emotions.columns = ['joy']
        mock_daily_emotions.index = [MagicMock()]

        mock_grouped = MagicMock()
        mock_filtered_df.groupby.return_value = mock_grouped

        mock_grouped.__getitem__.return_value.apply.return_value.unstack.return_value.resample.return_value.sum.return_value = mock_daily_emotions

        result = analyze_emotion_over_time(mock_df, username="Alice")

        self.assertEqual(result, "rendered_chartjs_html_user")
        mock_filter_by_user.assert_called_with(mock_df, "Alice")
