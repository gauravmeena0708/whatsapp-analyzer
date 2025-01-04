# tests/test_parser.py

import unittest
from whatsapp_analyzer.parser import WhatsAppParser
from whatsapp_analyzer.exceptions import ChatParseException
from datetime import datetime

class TestWhatsAppParser(unittest.TestCase):
    def test_parse_valid_line(self):
        parser = WhatsAppParser("dummy_file.txt")  # Filepath doesn't matter for this test
        line = "12/25/23, 10:00 AM - User1: Hello!"
        parser._parse_line(line)
        self.assertEqual(len(parser.data), 1)
        self.assertEqual(parser.data[0]["sender"], "User1")
        self.assertEqual(parser.data[0]["message"], "Hello!")

    def test_parse_invalid_line(self):
        parser = WhatsAppParser("dummy_file.txt")
        line = "This is not a valid chat line"
        parser._parse_line(line)
        self.assertEqual(len(parser.data), 0)  # No data should be added

    def test_parse_file_not_found(self):
        parser = WhatsAppParser("nonexistent_file.txt")
        with self.assertRaises(ChatParseException):
            parser.parse()
    
    def test_parse_valid_line_edge_cases(self):
        parser = WhatsAppParser("dummy_file.txt")
        # Test different date/time formats
        line1 = "12/25/2023, 10:00 AM - User1: Message1"
        line2 = "25/12/23, 10:00 PM - User2: Message2"
        line3 = "2023/12/25, 10:00 PM - User3: Message3"
        line4 = "12/25/23, 10:00:00 AM - User4: Message4"
        line5 = "12/25/23, 10:00 - User5: Message5"
        # ... add more variations of lines to test

        parser._parse_line(line1)
        parser._parse_line(line2)
        parser._parse_line(line3)
        parser._parse_line(line4)
        parser._parse_line(line5)
        # ... more calls to _parse_line

        self.assertEqual(len(parser.data), 5)
        self.assertEqual(parser.data[0]["date_time"], datetime(2023, 12, 25, 10, 0))
        self.assertEqual(parser.data[1]["date_time"], datetime(2023, 12, 25, 22, 0))
        self.assertEqual(parser.data[2]["date_time"], datetime(2023, 12, 25, 22, 0))
        self.assertEqual(parser.data[3]["date_time"], datetime(2023, 12, 25, 10, 0, 0))
        self.assertEqual(parser.data[4]["date_time"], datetime(2023, 12, 25, 10, 0))
        # ... add assertions for other lines

if __name__ == "__main__":
    unittest.main()