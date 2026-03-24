# tests/test_plot_utils.py

import unittest
from unittest.mock import MagicMock
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
from whatsapp_analyzer.plot_utils import clean_message

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

if __name__ == '__main__':
    unittest.main()
