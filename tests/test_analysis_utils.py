# tests/test_analysis_utils.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile

from whatsapp_analyzer.analysis_utils import analyze_message_timing, basic_stats
from whatsapp_analyzer.utils import df_basic_cleanup
from whatsapp_analyzer.parser import Parser
# For basic_stats, TextBlob is an indirect dependency for sentiment.
# NLTK resources (punkt, stopwords, vader_lexicon, averaged_perceptron_tagger, wordnet)
# are also expected to be available for basic_stats to run fully.

class TestAnalysisUtils(unittest.TestCase):

    def setUp(self):
        self.temp_file_handle, self.temp_file_path = tempfile.mkstemp(suffix=".txt", text=True)
        os.close(self.temp_file_handle)

    def tearDown(self):
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def _create_and_parse_chat_file(self, chat_lines_list_of_strings):
        """Helper to create a chat file, parse it, and run df_basic_cleanup."""
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            for line in chat_lines_list_of_strings:
                f.write(line + "\n")
        parser = Parser(self.temp_file_path)
        raw_df = parser.parse_chat_data()
        self.assertFalse(raw_df.empty, "Parser returned an empty DataFrame from test data.")
        # 't' column must exist for df_basic_cleanup
        self.assertIn('t', raw_df.columns, "Raw DataFrame must contain 't' column.")
        return df_basic_cleanup(raw_df.copy())

    # --- Tests for analyze_message_timing ---

    def test_analyze_message_timing_single_user(self):
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Message 1",
            "10/10/22, 10:05 AM - User1: Message 2", # 5 min diff
            "10/10/22, 10:00 AM - User2: U2 Message 1",
            "10/10/22, 10:10 AM - User1: Message 3", # 5 min diff from User1's last
            "10/10/22, 10:07 AM - User2: U2 Message 2"  # 7 min diff
        ]
        df = self._create_and_parse_chat_file(chat_lines)
        response_times_user1 = analyze_message_timing(df, username='User1')
        
        self.assertIsInstance(response_times_user1, pd.Series)
        self.assertEqual(len(response_times_user1), 2)
        pd.testing.assert_series_equal(response_times_user1, pd.Series([5.0, 5.0]), check_names=False, check_index=False)

    def test_analyze_message_timing_all_users(self):
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Message 1",
            "10/10/22, 10:05 AM - User1: Message 2", # User1: 5 min
            "10/10/22, 10:00 AM - User2: U2 Message 1",
            "10/10/22, 10:07 AM - User2: U2 Message 2", # User2: 7 min
            "10/10/22, 10:10 AM - User1: Message 3"  # User1: 5 min
        ]
        df = self._create_and_parse_chat_file(chat_lines)
        response_times_all = analyze_message_timing(df) # No username
        
        self.assertIsInstance(response_times_all, pd.Series)
        # Expecting 2 from User1, 1 from User2
        self.assertEqual(len(response_times_all), 3) 
        # Values will be [5.0, 5.0, 7.0] but order depends on groupby-diff-dropna sequence
        expected_values = sorted([5.0, 5.0, 7.0])
        actual_values = sorted(response_times_all.tolist())
        self.assertEqual(actual_values, expected_values)


    def test_analyze_message_timing_one_message_user(self):
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Only one message",
            "10/10/22, 10:05 AM - User2: U2 Message 1",
            "10/10/22, 10:10 AM - User2: U2 Message 2" 
        ]
        df = self._create_and_parse_chat_file(chat_lines)
        response_times_user1 = analyze_message_timing(df, username='User1')
        
        self.assertIsInstance(response_times_user1, pd.Series)
        self.assertTrue(response_times_user1.empty)

    def test_analyze_message_timing_known_gaps(self):
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Msg1",
            "10/10/22, 10:05 AM - User1: Msg2", # 5 min after Msg1
            "10/10/22, 10:20 AM - User1: Msg3"  # 15 min after Msg2
        ]
        df = self._create_and_parse_chat_file(chat_lines)
        response_times_user1 = analyze_message_timing(df, username='User1')
        
        self.assertIsInstance(response_times_user1, pd.Series)
        self.assertEqual(len(response_times_user1), 2)
        pd.testing.assert_series_equal(response_times_user1, pd.Series([5.0, 15.0]), check_names=False, check_index=False)

    # --- Tests for basic_stats (selected checks) ---

    def test_basic_stats_core_metrics(self):
        chat_lines = [
            "10/10/22, 09:00 AM - User1: Good morning! Feeling happy today. 😊",
            "10/10/22, 09:05 AM - User1: Let's make this a great day.",
            "10/10/22, 09:10 AM - User2: Sad news from my side.",
            "10/10/22, 09:15 AM - User1: Oh no, what happened? Bad news is not good."
        ]
        clean_df = self._create_and_parse_chat_file(chat_lines)
        # Ensure 'User1' exists in the parsed data
        self.assertTrue("User1" in clean_df['name'].unique(), "User1 should be in the DataFrame")

        stats_dict_user1 = basic_stats(clean_df.copy(), username='User1')

        self.assertEqual(stats_dict_user1['Total Messages'], 3)
        # "Good morning! Feeling happy today. 😊" -> 5 words (emoji not counted as word by split)
        # "Let's make this a great day." -> 7 words
        # "Oh no, what happened? Bad news is not good." -> 9 words
        self.assertEqual(stats_dict_user1['Total Words'], 5 + 7 + 9) 
        
        # Sentiment: "happy" positive, "good" positive. "sad" negative, "bad" negative.
        # User1 messages:
        # 1: "Good morning! Feeling happy today. 😊" (Positive due to "Good", "happy")
        # 2: "Let's make this a great day." (Positive due to "great")
        # 3: "Oh no, what happened? Bad news is not good." (Negative due to "Bad", "not good")
        # Note: TextBlob's default sentiment might vary. These are simple checks.
        # Requires TextBlob to be installed and vader_lexicon for TextBlob if not already downloaded by NLTK.
        self.assertGreaterEqual(stats_dict_user1['Positive Messages'], 2, "Expected at least 2 positive messages for User1")
        self.assertGreaterEqual(stats_dict_user1['Negative Messages'], 1, "Expected at least 1 negative message for User1")

        # Unique words count for User1:
        # "good", "morning", "feeling", "happy", "today", "let's", "make", "this", "a", "great", "day",
        # "oh", "no", "what", "happened", "bad", "news", "is", "not" (stopwords are removed by CountVectorizer in basic_stats)
        # This needs careful checking against how basic_stats calculates unique_words_count (stopwords list etc.)
        # For now, just check it's a plausible number (positive integer)
        self.assertIsInstance(stats_dict_user1['Unique Words Count'], int)
        self.assertGreater(stats_dict_user1['Unique Words Count'], 5) # Conservative check


    def test_basic_stats_formatted_strings(self):
        chat_lines = [
            "10/10/22, 09:00 AM - User1: test one two",
            "10/10/22, 09:05 AM - User1: test three"
        ]
        clean_df = self._create_and_parse_chat_file(chat_lines)
        stats_dict_user1 = basic_stats(clean_df.copy(), username='User1')

        self.assertIsInstance(stats_dict_user1['Common Unigrams'], str)
        self.assertTrue("<li>test: 2</li>" in stats_dict_user1['Common Unigrams'] or "<li>test: 2</li>" in stats_dict_user1['Common Unigrams'])
        self.assertIsInstance(stats_dict_user1['Hindi Abuse Counts'], str)
        self.assertEqual(stats_dict_user1['Hindi Abuse Counts'], "<ul></ul>") # Assuming no abuse words

    def test_basic_stats_runs_without_error(self):
        """ Test that basic_stats completes without error for a typical case."""
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Hello world http://example.com 😊",
            "10/10/22, 10:05 AM - User1: Second message",
            "10/11/22, 11:30 AM - User2: <Media omitted>",
            "10/12/22, 12:20 PM - User3: Test message with emoji 😱 and url www.test.com",
            "10/13/22, 01:25 PM - User1: <This message was edited>",
            "10/14/22, 02:30 PM - User2: This message was deleted"
        ]
        clean_df = self._create_and_parse_chat_file(chat_lines)
        
        try:
            # Test for a specific user
            if "User1" in clean_df['name'].unique():
                 basic_stats(clean_df.copy(), username='User1')
            # Test for all users (username=None)
            basic_stats(clean_df.copy()) 
        except Exception as e:
            self.fail(f"basic_stats raised an exception unexpectedly: {e}")

if __name__ == '__main__':
    unittest.main()
