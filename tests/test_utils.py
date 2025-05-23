# tests/test_utils.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import os
import tempfile

from whatsapp_analyzer.utils import df_basic_cleanup
from whatsapp_analyzer.parser import Parser

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Create a temporary file path for chat data
        self.temp_file_handle, self.temp_file_path = tempfile.mkstemp(suffix=".txt", text=True)
        os.close(self.temp_file_handle) # Close the handle, we'll open/write/close as needed in tests

    def tearDown(self):
        # Clean up the temporary file
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def _create_chat_file(self, content_lines):
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            for line in content_lines:
                f.write(line + "\n") # Ensure each entry is a line
        return self.temp_file_path

    def test_df_basic_cleanup_columns_exist(self):
        """Test that all expected columns are created by df_basic_cleanup."""
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Hello world",
            "10/11/22, 11:30 AM - User2: <Media omitted>"
        ]
        chat_file = self._create_chat_file(chat_lines)
        
        parser = Parser(chat_file)
        raw_df = parser.parse_chat_data()
        
        # Ensure raw_df is not empty and has 't' column before cleanup
        self.assertFalse(raw_df.empty, "Raw DataFrame should not be empty")
        self.assertIn('t', raw_df.columns, "'t' column must be in raw_df for cleanup")

        clean_df = df_basic_cleanup(raw_df.copy())

        expected_columns = [
            'date_time', 'date', 'year', 'month', 'monthn', 'day', 'dayn', 
            'woy', 'doy', 'dow', 'ym', 'yw', 'yd', 'md', 'time', 'hour', 
            'min', 'hm', 'name', 'message', 'mlen', 'emoji', 'emojicount', 
            'urls', 'urlcount', 'yturls', 'yturlcount', 'mediacount', 
            'editcount', 'deletecount'
        ]
        
        missing_columns = [col for col in expected_columns if col not in clean_df.columns]
        self.assertEqual(len(missing_columns), 0, f"Missing columns: {missing_columns}")
        
        extra_columns = [col for col in clean_df.columns if col not in expected_columns]
        self.assertEqual(len(extra_columns), 0, f"Extra columns found: {extra_columns}")


    def test_df_basic_cleanup_data_types(self):
        """Test data types of key columns after df_basic_cleanup."""
        chat_lines = [
            "10/10/22, 10:00 AM - User1: Test message http://example.com 😊",
            "10/11/22, 11:30 AM - User2: <Media omitted>"
        ]
        chat_file = self._create_chat_file(chat_lines)
        parser = Parser(chat_file)
        raw_df = parser.parse_chat_data()
        clean_df = df_basic_cleanup(raw_df.copy())

        self.assertFalse(clean_df.empty, "Cleaned DataFrame should not be empty for type checking")

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(clean_df['date_time']), "date_time column should be datetime type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['emojicount']), "emojicount column should be integer type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['urlcount']), "urlcount column should be integer type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['hour']), "hour column should be integer type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['mlen']), "mlen column should be integer type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['mediacount']), "mediacount column should be integer type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['editcount']), "editcount column should be integer type")
        self.assertTrue(pd.api.types.is_integer_dtype(clean_df['deletecount']), "deletecount column should be integer type")
        self.assertEqual(clean_df['date'].dtype, np.dtype('object')) # pandas stores date objects as 'object'
        self.assertTrue(isinstance(clean_df['date'].iloc[0], date), "date column elements should be datetime.date")


    def test_df_basic_cleanup_calculated_values(self):
        """Test specific calculated values from df_basic_cleanup."""
        # Note: 10/10/2022 was a Monday
        chat_lines = [
            "10/10/22, 10:05 AM - User1: Hello world http://example.com 😊",
            "10/11/22, 11:15 AM - User2: <Media omitted>",
            "10/12/22, 12:20 PM - User1: This message was edited", # This is not the WhatsApp system message for edits
            "10/13/22, 01:25 PM - User1: <This message was edited>", # This IS the system message
            "10/14/22, 02:30 PM - User2: This message was deleted", # This IS a system message for deletion
            "10/15/22, 03:35 PM - User1: You deleted this message" # This IS also a system message for deletion
        ]
        chat_file = self._create_chat_file(chat_lines)
        parser = Parser(chat_file)
        raw_df = parser.parse_chat_data()
        clean_df = df_basic_cleanup(raw_df.copy())

        self.assertEqual(len(clean_df), 6)

        # Test first message (User1)
        msg1 = clean_df.iloc[0]
        self.assertEqual(msg1['name'], "User1")
        self.assertEqual(msg1['message'], "Hello world http://example.com 😊")
        self.assertEqual(msg1['mlen'], len("Hello world http://example.com 😊"))
        self.assertEqual(msg1['emojicount'], 1)
        self.assertEqual(msg1['urlcount'], 1)
        self.assertEqual(msg1['mediacount'], 0)
        self.assertEqual(msg1['editcount'], 0)
        self.assertEqual(msg1['deletecount'], 0)
        self.assertEqual(msg1['hour'], 10)
        self.assertEqual(msg1['dayn'], 'Monday')
        self.assertEqual(msg1['monthn'], 'October')
        self.assertEqual(msg1['year'], 2022)
        self.assertEqual(msg1['hm'], 10.08) # 10 + 5/60

        # Test second message (User2) - Media
        msg2 = clean_df.iloc[1]
        self.assertEqual(msg2['name'], "User2")
        self.assertEqual(msg2['message'], "<Media omitted>")
        self.assertEqual(msg2['mediacount'], 1)

        # Test third message (User1) - Contains "This message was edited" but NOT the system way
        msg3 = clean_df.iloc[2]
        self.assertEqual(msg3['editcount'], 0) 

        # Test fourth message (User1) - Actual edited message
        msg4 = clean_df.iloc[3]
        self.assertEqual(msg4['editcount'], 1)

        # Test fifth message (User2) - "This message was deleted"
        msg5 = clean_df.iloc[4]
        self.assertEqual(msg5['deletecount'], 1)
        
        # Test sixth message (User1) - "You deleted this message"
        msg6 = clean_df.iloc[5]
        self.assertEqual(msg6['deletecount'], 1)

if __name__ == '__main__':
    unittest.main()
