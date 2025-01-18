# whatsapp_analyzer/stats.py

from statistics import mean, median, stdev
from collections import Counter

def calculate_message_lengths(message_lengths):
    """
    Calculates basic statistics on message lengths.
    """
    if not message_lengths:
        return {
            "mean": 0,
            "median": 0,
            "std_dev": 0,
            "max": 0,
            "min": 0
        }

    return {
        "mean": mean(message_lengths),
        "median": median(message_lengths),
        "std_dev": stdev(message_lengths),
        "max": max(message_lengths),
        "min": min(message_lengths),
    }

def calculate_active_periods(chat_data, n=5):
    """
    Finds the N most active time periods (e.g., days or hours).

    Args:
        chat_data: List of parsed chat messages.
        n: Number of top active periods to return.

    Returns:
        List of tuples: (time_period, message_count)
    """
    time_periods = Counter()
    for msg in chat_data:
        # Example: Count activity by hour
        if msg["sender"] != "System":
          time_periods[msg["date_time"].hour] += 1 

    return time_periods.most_common(n)