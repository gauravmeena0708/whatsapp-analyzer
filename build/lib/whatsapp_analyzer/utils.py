# whatsapp_analyzer/utils.py
import regex
import emoji
import pandas as pd
import numpy as np

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

def df_basic_cleanup(df):
    """
    Performs basic data cleaning and feature engineering on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing parsed chat data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with added features.
    """
    df["date_time"] = pd.to_datetime(df["t"])
    df["date"] = df["date_time"].dt.date
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month.astype(str).str.zfill(2)
    df["day"] = df["date_time"].dt.day

    df["dayn"] = df["date_time"].dt.day_name().astype("category")
    df["monthn"] = df["date_time"].dt.month_name()

    df["doy"] = df["date_time"].dt.dayofyear
    df["dow"] = df["date_time"].dt.dayofweek
    df["woy"] = df["date_time"].dt.isocalendar().week
    df["time"] = df["date_time"].dt.time
    df["hour"] = df["date_time"].dt.hour
    df["min"] = df["date_time"].dt.minute
    df["hm"] = df["hour"] + round(df["min"] / 60, 2)

    df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    df["yw"] = df["year"].astype(str) + "-" + df["woy"].astype(str)
    df["yd"] = df["year"].astype(str) + "-" + df["doy"].astype(str)
    df["md"] = df["monthn"].astype(str) + "-" + df["date"].astype(str)

    df["mlen"] = df["message"].str.len()

    df["emoji"] = df["message"].apply(get_emojis)
    df["emojicount"] = df["emoji"].str.len()

    df["urls"] = df["message"].apply(get_urls)
    df["urlcount"] = df["urls"].str.len()

    df["yturls"] = df["message"].apply(get_yturls)
    df["yturlcount"] = df["yturls"].str.len()

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

    df.drop("t", inplace=True, axis=1)
    df = df[
        [
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
    ]
    return df