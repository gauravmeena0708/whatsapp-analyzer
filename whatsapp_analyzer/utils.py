# whatsapp_analyzer/utils.py
import regex
import emoji
import pandas as pd
import numpy as np
from .constants import skill_keywords, hindi_abusive_words, stop_words

URL_PATTERN = r"(https?://\S+)"
YOUTUBE_PATTERN = r"(https?://youtu(\.be|be\.com)\S+)"

def get_emojis(text):
    emoji_list = []
    data = regex.findall(r"\X", text)
    for word in data:
        if any(char in emoji.EMOJI_DATA for char in word):
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
    # The extra ']' was here, it has been removed.
    # Filter out any columns not in the original DataFrame to handle cases where some columns might be missing
    # This also ensures that if new columns are added by helpers but not in final_columns_order, they are dropped.
    existing_columns = [col for col in final_columns_order if col in df.columns]
    df = df[existing_columns]
    
    return df