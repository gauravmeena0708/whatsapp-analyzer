# tests/test_parser.py

import unittest
import pandas as pd
from datetime import datetime
import os
import tempfile
from whatsapp_analyzer.parser import Parser

class TestParser(unittest.TestCase):

    def setUp(self):
        # Create a temporary file that can be used by multiple tests needing file operations
        self.temp_file_handle, self.temp_file_path = tempfile.mkstemp(suffix=".txt", text=True)
        # Close the handle immediately so it can be opened by other processes or modes
        os.close(self.temp_file_handle)

    def tearDown(self):
        # Clean up the temporary file
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    # --- Tests for preprocess_lines ---
    def test_preprocess_single_lines(self):
        parser_instance = Parser("dummy.txt") # File path not used for this direct method call
        lines = [
            "10/10/22, 10:00 AM - User1: Message 1\n",
            "10/10/22, 10:01 AM - User2: Message 2\n",
            "10/10/22, 10:02 AM - User1: Message 3\n"
        ]
        processed = parser_instance.preprocess_lines(lines)
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0], "10/10/22, 10:00 AM - User1: Message 1")
        self.assertEqual(processed[1], "10/10/22, 10:01 AM - User2: Message 2")
        self.assertEqual(processed[2], "10/10/22, 10:02 AM - User1: Message 3")

    def test_preprocess_multiline_messages(self):
        parser_instance = Parser("dummy.txt")
        lines = [
            "10/10/22, 10:00 AM - User1: This is the first line\n",
            "of a multiline message.\n",
            "10/10/22, 10:01 AM - User2: This is a single line message.\n",
            "10/10/22, 10:02 AM - User1: Another multiline\n",
            "message here\n",
            "and here.\n"
        ]
        processed = parser_instance.preprocess_lines(lines)
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0], "10/10/22, 10:00 AM - User1: This is the first line of a multiline message.")
        self.assertEqual(processed[1], "10/10/22, 10:01 AM - User2: This is a single line message.")
        self.assertEqual(processed[2], "10/10/22, 10:02 AM - User1: Another multiline message here and here.")

    def test_preprocess_empty_input(self):
        parser_instance = Parser("dummy.txt")
        lines = []
        processed = parser_instance.preprocess_lines(lines)
        self.assertEqual(len(processed), 0)

    # --- Tests for parse_chat_data ---
    def test_parse_chat_data_simple(self):
        chat_content = [
            "10/10/22, 10:00 AM - User1: Hello there!\n",
            "10/10/22, 10:01 AM - User2: Hi User1.\n",
            "10/10/22, 10:02 AM - User2 changed this group's icon\n" 
        ]
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.writelines(chat_content)

        parser_instance = Parser(self.temp_file_path)
        df = parser_instance.parse_chat_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        
        self.assertEqual(df.iloc[0]['t'], datetime(2022, 10, 10, 10, 0, 0))
        self.assertEqual(df.iloc[0]['name'], "User1")
        self.assertEqual(df.iloc[0]['message'], "Hello there!")

        self.assertEqual(df.iloc[1]['t'], datetime(2022, 10, 10, 10, 1, 0))
        self.assertEqual(df.iloc[1]['name'], "User2")
        self.assertEqual(df.iloc[1]['message'], "Hi User1.")
        
        self.assertEqual(df.iloc[2]['t'], datetime(2022, 10, 10, 10, 2, 0))
        self.assertEqual(df.iloc[2]['name'], "System") 
        self.assertEqual(df.iloc[2]['message'], "User2 changed this group's icon")


    def test_parse_chat_data_multiline(self):
        chat_content = [
            "10/10/22, 11:00 AM - UserA: First part of message.\n",
            "Second part of message.\n",
            "Third part.\n",
            "10/10/22, 11:05 AM - UserB: Another message.\n"
        ]
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.writelines(chat_content)
        
        parser_instance = Parser(self.temp_file_path)
        df = parser_instance.parse_chat_data()

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['message'], "First part of message. Second part of message. Third part.")
        self.assertEqual(df.iloc[0]['name'], "UserA")
        self.assertEqual(df.iloc[1]['name'], "UserB")

    def test_parse_chat_data_system_messages(self):
        chat_content = [
            "01/01/23, 12:01 AM - User1 joined using this group's invite link\n",
            "01/01/23, 12:02 AM - You created group \"Test Group\"\n",
            "01/01/23, 12:03 AM - User2: Hello all!\n"
        ]
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.writelines(chat_content)

        parser_instance = Parser(self.temp_file_path)
        df = parser_instance.parse_chat_data()
        
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]['name'], "System")
        self.assertEqual(df.iloc[0]['message'], "User1 joined using this group's invite link")
        self.assertEqual(df.iloc[1]['name'], "System")
        self.assertEqual(df.iloc[1]['message'], "You created group \"Test Group\"")
        self.assertEqual(df.iloc[2]['name'], "User2")
        self.assertEqual(df.iloc[2]['message'], "Hello all!")

    def test_parse_chat_data_invalid_lines(self):
        chat_content = [
            "This is a completely invalid line.\n", 
            "10/10/22, 10:00 AM - User1: Valid message 1.\n",
            "Another invalid line, no timestamp.\n", 
            "10/10/22, 10:01 AM - User2: Valid message 2.\n",
            "This line belongs to message 2.\n", 
            "Yet another invalid line.\n" 
        ]
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.writelines(chat_content)

        parser_instance = Parser(self.temp_file_path)
        df = parser_instance.parse_chat_data()
        
        self.assertEqual(len(df), 2) 
        self.assertEqual(df.iloc[0]['message'], "Valid message 1. Another invalid line, no timestamp.")
        self.assertEqual(df.iloc[1]['message'], "Valid message 2. This line belongs to message 2. Yet another invalid line.")


    def test_parse_chat_data_file_not_found(self):
        parser_instance = Parser("non_existent_file.txt")
        with self.assertRaises(FileNotFoundError):
            parser_instance.parse_chat_data()

    def test_parse_chat_data_empty_file(self):
        with open(self.temp_file_path, 'w', encoding='utf-8') as f:
            pass 
        
        parser_instance = Parser(self.temp_file_path)
        df = parser_instance.parse_chat_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

if __name__ == "__main__":
    unittest.main()