# tests/test_utils.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import os
import tempfile

from whatsapp_analyzer.utils import df_basic_cleanup, anonymize, ANIMAL_NAMES
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

# --- TestAnonymize class starts here ---
class TestAnonymize(unittest.TestCase):

    def setUp(self):
        # Create temporary files for input and output
        self.temp_input_file = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
        self.temp_output_file = tempfile.NamedTemporaryFile(mode="r", delete=False, encoding="utf-8")
        
        # Store their paths
        self.input_path = self.temp_input_file.name
        self.output_path = self.temp_output_file.name
        
        # Close them immediately, anonymize function will open/close them
        self.temp_input_file.close()
        self.temp_output_file.close()

    def tearDown(self):
        # Clean up the temporary files
        os.remove(self.input_path)
        os.remove(self.output_path)

    def _write_to_input(self, lines):
        with open(self.input_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    def _read_file_lines(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            return [] # Return empty list if file not found (e.g. for not_found test)

    def test_basic_anonymization(self):
        input_lines = [
            "20/03/2023, 10:00 - Alice: Hello Bob",
            "20/03/2023, 10:01 - Bob: Hi Alice",
            "20/03/2023, 10:02 - Alice: How are you?"
        ]
        self._write_to_input(input_lines)
        anonymize(self.input_path, self.output_path)
        output_lines = self._read_file_lines(self.output_path)
        
        animal1 = ANIMAL_NAMES[0]
        animal2 = ANIMAL_NAMES[1]
        
        expected_lines = [
            f"20/03/2023, 10:00 - user_1_{animal1}: Hello Bob",
            f"20/03/2023, 10:01 - user_2_{animal2}: Hi Alice",
            f"20/03/2023, 10:02 - user_1_{animal1}: How are you?"
        ]
        self.assertEqual(output_lines, expected_lines)

    def test_system_messages(self):
        input_lines = [
            "20/03/2023, 10:00 - System: Alice added Bob",
            "20/03/2023, 10:01 - Carol: Hello everyone",
            "This is a line that won't be parsed by _parse_line."
        ]
        self._write_to_input(input_lines)
        anonymize(self.input_path, self.output_path)
        output_lines = self._read_file_lines(self.output_path)
        
        animal1 = ANIMAL_NAMES[0] # Carol will be user_1

        expected_lines = [
            "20/03/2023, 10:00 - System: Alice added Bob",
            f"20/03/2023, 10:01 - user_1_{animal1}: Hello everyone",
            "This is a line that won't be parsed by _parse_line."
        ]
        self.assertEqual(output_lines, expected_lines)

    def test_username_in_message_content(self):
        input_lines = [
            "20/03/2023, 10:00 - Dave: Hey Eve, how is Dave doing?",
            "20/03/2023, 10:01 - Eve: Dave is fine."
        ]
        self._write_to_input(input_lines)
        anonymize(self.input_path, self.output_path)
        output_lines = self._read_file_lines(self.output_path)

        animal1 = ANIMAL_NAMES[0] # Dave
        animal2 = ANIMAL_NAMES[1] # Eve
        
        expected_lines = [
            f"20/03/2023, 10:00 - user_1_{animal1}: Hey Eve, how is Dave doing?",
            f"20/03/2023, 10:01 - user_2_{animal2}: Dave is fine."
        ]
        self.assertEqual(output_lines, expected_lines)

    def test_empty_input_file(self):
        self._write_to_input([])
        anonymize(self.input_path, self.output_path)
        output_lines = self._read_file_lines(self.output_path)
        self.assertEqual(output_lines, [])

    def test_no_matching_chat_lines(self):
        input_lines = [
            "This is just some random text.",
            "Another line without WhatsApp format.",
            "12345"
        ]
        self._write_to_input(input_lines)
        anonymize(self.input_path, self.output_path)
        output_lines = self._read_file_lines(self.output_path)
        # Expect output to be same as input, plus newline handling (strip ensures comparison is fair)
        self.assertEqual(output_lines, input_lines)

    def test_animal_list_cycling(self):
        num_users = len(ANIMAL_NAMES) + 2  # e.g., 12 users if 10 animals
        input_lines = []
        expected_lines = []
        for i in range(num_users):
            user_name = f"User{i+1}"
            input_lines.append(f"20/03/2023, 10:{i:02d} - {user_name}: Message {i+1}")
            
            animal_index = i % len(ANIMAL_NAMES)
            expected_animal = ANIMAL_NAMES[animal_index]
            expected_lines.append(f"20/03/2023, 10:{i:02d} - user_{i+1}_{expected_animal}: Message {i+1}")
            
        self._write_to_input(input_lines)
        anonymize(self.input_path, self.output_path)
        output_lines = self._read_file_lines(self.output_path)
        self.assertEqual(output_lines, expected_lines)

    def test_input_file_not_found(self):
        # Ensure the input file does not exist
        non_existent_input_path = "/tmp/this_file_should_definitely_not_exist_for_test.txt"
        if os.path.exists(non_existent_input_path):
             os.remove(non_existent_input_path) # Just in case

        # Suppress print output from anonymize for this test
        import sys
        from io import StringIO
        original_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        anonymize(non_existent_input_path, self.output_path)
        
        sys.stdout = original_stdout # Restore stdout
        
        # Check that an error message was printed (optional, but good to confirm)
        # self.assertIn("Error: Input file", captured_output.getvalue())

        # Check that the output file was not created or is empty
        output_lines = self._read_file_lines(self.output_path)
        self.assertEqual(output_lines, [], "Output file should be empty or not created if input is not found.")


if __name__ == '__main__':
    unittest.main()
