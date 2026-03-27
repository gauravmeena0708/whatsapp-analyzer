import re
from dateutil import parser
import pandas as pd

class Parser:
    def __init__(self, file_path):
        self.file_path = file_path

    def preprocess_lines(self, lines):
        processed_lines = []
        buffer = ""
        for line in lines:
            if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4},", line):
                if buffer:
                    processed_lines.append(buffer)
                buffer = line.strip()
            else:
                if buffer:
                    buffer += f" {line.strip()}"
                else:
                    buffer = line.strip()
        if buffer:
            processed_lines.append(buffer)
        return processed_lines

    def infer_dayfirst(self, lines):
        dayfirst_votes = 0
        monthfirst_votes = 0

        for line in lines[:200]:
            match = re.match(r"^(\d{1,2})/(\d{1,2})/\d{2,4},", line)
            if not match:
                continue

            first, second = map(int, match.groups())
            if first > 12 and second <= 12:
                dayfirst_votes += 1
            elif second > 12 and first <= 12:
                monthfirst_votes += 1

        if monthfirst_votes > dayfirst_votes:
            return False
        return True

    def parse_chat_data(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            chat_lines = file.readlines()

        chat_lines = self.preprocess_lines(chat_lines)
        dayfirst = self.infer_dayfirst(chat_lines)

        parsed_rows = []
        for line in chat_lines:
            # Captures: date, time, sender, message
            match = re.match(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s*-\s*([^:]+?):\s*(.*)", line)
            if match:
                d_str, t_str, sender, msg = match.groups()
                try:
                    dt = parser.parse(f"{d_str} {t_str}", dayfirst=dayfirst)
                    parsed_rows.append([dt, sender, msg])
                except:
                    continue
            else:
                # Try system message
                match2 = re.match(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s*-\s*(.*)", line)
                if match2:
                    d_str, t_str, event = match2.groups()
                    try:
                        dt = parser.parse(f"{d_str} {t_str}", dayfirst=dayfirst)
                        parsed_rows.append([dt, "System", event])
                    except:
                        continue
        
        if not parsed_rows:
            return pd.DataFrame(columns=["t", "name", "message"])
            
        return pd.DataFrame(parsed_rows, columns=["t", "name", "message"])
