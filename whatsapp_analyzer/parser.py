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

    def preprocess_lines(self, lines):
        """
        Preprocesses the chat lines to remove line breaks if the next line
        does not start with a valid date pattern.

        Args:
            lines (list): List of chat lines.

        Returns:
            list: Preprocessed chat lines.
        """
        processed_lines = []
        buffer = ""

        # Updated date_pattern
        date_pattern = re.compile(r"^\d{2}/\d{2}/\d{2,4}, \d{1,2}:\d{2}\s(?:AM|PM|am|pm)")

        for line in lines:
            if date_pattern.match(line):
                if buffer:
                    processed_lines.append(buffer)
                buffer = line.strip()
            else:
                buffer += f" {line.strip()}"

        if buffer:  # Add the last buffered line
            processed_lines.append(buffer)

        return processed_lines

    def parse_chat_data(self):
        """Parses the WhatsApp chat file and returns a DataFrame."""
        with open(self.file_path, "r", encoding="utf-8") as file:
            chat_lines = file.readlines()

        # Preprocess lines to handle multiline messages
        chat_lines = self.preprocess_lines(chat_lines)

        chat_data = []
        for line in chat_lines:
            try:
                # Updated user message regex
                match = re.match(
                    r"(\d{2}/\d{2}/\d{2,4}, \d{1,2}:\d{2}\s(?:AM|PM|am|pm)) - (.*?): (.*)", line
                )
                if match:
                    timestamp, sender, message = match.groups()
                    date_obj = parser.parse(timestamp)
                    chat_data.append({"t": date_obj, "name": sender, "message": message})
                else:
                    # Updated system message regex
                    match2 = re.match(
                        r"(\d{2}/\d{2}/\d{2,4}, \d{1,2}:\d{2}\s(?:AM|PM|am|pm)) - (.*)", line
                    )
                    if match2:
                        timestamp, event = match2.groups()
                        date_obj = parser.parse(timestamp)
                        chat_data.append({"t": date_obj, "name": "System", "message": event})
                    else:
                        print(f"Skipping line: {line.strip()} (No match)")
            except (ValueError, AttributeError):
                print(f"Skipping line: {line.strip()} (Parse error)")

        return pd.DataFrame(chat_data)
