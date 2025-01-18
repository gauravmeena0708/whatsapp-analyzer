# whatsapp_analyzer/parser.py
import re
from dateutil import parser
import pandas as pd

class Parser:
    def __init__(self, file_path):
        """
        Initializes the Parser with the path to the WhatsApp chat file.

        Args:
            file_path (str): Path to the chat file.
        """
        self.file_path = file_path

    def parse_chat_data(self):
        """Parses the WhatsApp chat file and returns a DataFrame."""
        with open(self.file_path, "r", encoding="utf-8") as file:
            chat_lines = file.readlines()

        chat_data = []
        for line in chat_lines:
            try:
                match = re.match(
                    r"(\d{2}\/\d{2}\/\d{2}, \d{1,2}:\d{2} [apm]{2}) - (.*?): (.*)", line
                )
                if match:
                    timestamp, sender, message = match.groups()
                    date_obj = parser.parse(timestamp)
                    chat_data.append({"t": date_obj, "name": sender, "message": message})
            except (ValueError, AttributeError):
                print(f"Skipping line: {line.strip()} (Parse error)")

        return pd.DataFrame(chat_data)