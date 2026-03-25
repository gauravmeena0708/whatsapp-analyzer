# whatsapp_analyzer/utils.py
try:
    import regex
    HAS_REGEX_GRAPHEME = True
except ModuleNotFoundError:
    import re as regex
    HAS_REGEX_GRAPHEME = False

try:
    import emoji
    EMOJI_DATA = emoji.EMOJI_DATA
except ModuleNotFoundError:
    EMOJI_DATA = {}

import pandas as pd
import numpy as np
import re
import itertools

URL_PATTERN = r"(https?://\S+)"
YOUTUBE_PATTERN = r"(https?://youtu(\.be|be\.com)\S+)"

ANIMAL_NAMES = ["Panda", "Shark", "Zebra", "Lion", "Tiger", "Bear", "Eagle", "Wolf", "Fox", "Deer"]

def get_emojis(text):
    """Extracts emojis from text, handling multi-character emojis correctly if regex is available."""
    emoji_list = []
    data = regex.findall(r"\X", text) if HAS_REGEX_GRAPHEME else list(text)
    for word in data:
        if any(char in EMOJI_DATA or ord(char) > 10000 for char in word):
            emoji_list.append(word)
    return emoji_list

def get_urls(text):
    url_list = regex.findall(URL_PATTERN, text)
    return url_list

def get_yturls(text):
    url_list = regex.findall(YOUTUBE_PATTERN, text)
    return url_list

def _extract_datetime_features(df):
    """Extracts date and time based features from the 'date_time' column."""
    df["date"] = df["date_time"].dt.date
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month.astype(str).str.zfill(2)
    df["day"] = df["date_time"].dt.day

    df["dayn"] = df["date_time"].dt.day_name().astype("category")
    df["monthn"] = df["date_time"].dt.month_name()

    df["doy"] = df["date_time"].dt.dayofyear
    df["dow"] = df["date_time"].dt.dayofweek
    df["woy"] = df["date_time"].dt.isocalendar().week.astype(str) # Ensure woy is string for yw concatenation
    df["time"] = df["date_time"].dt.time
    df["hour"] = df["date_time"].dt.hour
    df["min"] = df["date_time"].dt.minute
    df["hm"] = df["hour"] + round(df["min"] / 60, 2)

    df["ym"] = df["year"].astype(str) + "-" + df["month"]
    df["yw"] = df["year"].astype(str) + "-" + df["woy"]
    df["yd"] = df["year"].astype(str) + "-" + df["doy"].astype(str).str.zfill(3) # Ensure doy is string and padded
    df["md"] = df["monthn"].astype(str) + "-" + df["day"].astype(str).str.zfill(2) # Ensure day is string and padded
    return df

def _extract_message_content_features(df):
    """Extracts features from the message content."""
    df["mlen"] = df["message"].str.len()
    df["emoji"] = df["message"].apply(get_emojis)
    df["emojicount"] = df["emoji"].str.len()
    df["urls"] = df["message"].apply(get_urls)
    df["urlcount"] = df["urls"].str.len()
    df["yturls"] = df["message"].apply(get_yturls)
    df["yturlcount"] = df["yturls"].str.len()
    return df

def _extract_message_type_counts(df):
    """Extracts counts of media, edited, and deleted messages."""
    df["mediacount"] = np.where(df["message"] == "<Media omitted>", 1, 0)
    df["editcount"] = np.where(
        df["message"].str.contains("<This message was edited>"), 1, 0
    )
    df["deletecount"] = np.where(
        (
            (df["message"] == "This message was deleted")
            | (df["message"] == "You deleted this message")
        ),
        1,
        0,
    )
    return df

def df_basic_cleanup(df):
    """
    Performs basic data cleaning and feature engineering on the DataFrame.
    Orchestrates calls to helper functions to extract features.
    """
    # Ensure 't' column exists and is the primary source for 'date_time'
    if 't' not in df.columns:
        raise ValueError("Input DataFrame must contain a 't' column for timestamp information.")
    
    df["date_time"] = pd.to_datetime(df["t"], errors='coerce')
    
    # Drop rows where 'date_time' could not be parsed, if any
    df.dropna(subset=['date_time'], inplace=True)

    df = _extract_datetime_features(df)
    df = _extract_message_content_features(df)
    df = _extract_message_type_counts(df)

    df.drop("t", inplace=True, axis=1)
    
    # Define the final order of columns
    final_columns_order = [
            "date_time",
            "date",
            "year",
            "month",
            "monthn",
            "day",
            "dayn",
            "woy",
            "doy",
            "dow",
            "ym",
            "yw",
            "yd",
            "md",
            "time",
            "hour",
            "min",
            "hm",
            "name",
            "message",
            "mlen",
            "emoji",
            "emojicount",
            "urls",
            "urlcount",
            "yturls",
            "yturlcount",
            "mediacount",
            "editcount",
            "deletecount",
        ]
    # Filter out any columns not in the original DataFrame to handle cases where some columns might be missing
    existing_columns = [col for col in final_columns_order if col in df.columns]
    df = df[existing_columns]
    
    return df

def anonymize(input_chat_path: str, output_chat_path: str):
    """Anonymizes usernames in a WhatsApp chat file and saves it to a new file."""
    username_to_anonymous_map = {}
    user_id_counter = 0
    animal_cycle = itertools.cycle(ANIMAL_NAMES)
    output_lines = []

    # Pattern for user messages: captures date, time, author, and message
    # e.g., "20/03/2023, 10:00 AM - Alice: Hello Bob"
    user_message_pattern = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?:\s*(?:AM|PM|am|pm))?)\s*-\s*([^:]+?):\s*(.*)"
    )

    try:
        with open(input_chat_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.rstrip("\n")
                if not stripped_line:
                    output_lines.append("")
                    continue

                user_match = user_message_pattern.match(stripped_line)
                if user_match:
                    date_str, time_str, author_str, message_str = user_match.groups()
                    author = author_str.strip()
                    if author.lower() != "system" and author not in username_to_anonymous_map:
                        user_id_counter += 1
                        anonymous_name = f"user_{user_id_counter}_{next(animal_cycle)}"
                        username_to_anonymous_map[author] = anonymous_name
                    display_author = username_to_anonymous_map.get(author, author)
                    output_lines.append(
                        f"{date_str.strip()}, {time_str.strip()} - {display_author}: {message_str.strip()}"
                    )
                else:
                    # Preserve system messages, events, and unparsed lines exactly.
                    output_lines.append(stripped_line)
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_chat_path}' not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    try:
        with open(output_chat_path, "w", encoding="utf-8") as f:
            for item in output_lines:
                f.write(item + '\n')
        print(f"Anonymized chat saved to {output_chat_path}")
    except IOError as e:
        print(f"Error writing to output file '{output_chat_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")
