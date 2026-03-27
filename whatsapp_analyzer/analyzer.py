# analyzer.py (inside whatsapp_analyzer)
import csv
import json
import os
import re
from collections import Counter, defaultdict
from html import escape
import pandas as pd
import numpy as np
from .parser import Parser
from .utils import (
    df_basic_cleanup,
)
from .plot_utils import plot_user_relationship_graph, render_chartjs, clean_message, extract_emojis
from .constants import stop_words
from .analysis_utils import basic_stats
from .constants import html_template, index_template
from .ml_models import get_sentence_embeddings, predict_sentiment


def _safe_float(value):
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value):
    try:
        if pd.isna(value):
            return 0
        return int(round(float(value)))
    except Exception:
        return 0


def _format_minutes(value):
    minutes = _safe_float(value)
    if minutes <= 0:
        return "N/A"
    if minutes < 60:
        return f"{minutes:.1f} mins"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hrs"
    days = hours / 24
    return f"{days:.1f} days"


def _build_html_table(title, headers, rows):
    if not rows:
        return f"""
        <div class="col-lg-6">
            <div class="mini-table-card h-100">
                <h4>{escape(title)}</h4>
                <p class="text-muted mb-0">No data available.</p>
            </div>
        </div>
        """

    head_html = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    row_html = ""
    for row in rows:
        row_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"

    return f"""
    <div class="col-lg-6">
        <div class="mini-table-card h-100">
            <h4>{escape(title)}</h4>
            <div class="table-responsive">
                <table class="table table-sm table-striped align-middle mb-0">
                    <thead><tr>{head_html}</tr></thead>
                    <tbody>{row_html}</tbody>
                </table>
            </div>
        </div>
    </div>
    """


def _normalize_birthday_date(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    for fmt in ("%m-%d", "%d-%m", "%Y-%m-%d", "%d/%m", "%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b", "%b-%d", "%B-%d", "%d %b", "%d %B"):
        parsed = pd.to_datetime(value, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed.strftime("%b %d")
    parsed = pd.to_datetime(value, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%b %d")
    return None


def _load_manual_birthdays(chat_file, out_dir, participants):
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(chat_file)), "birthdays.json"),
        os.path.join(os.path.dirname(os.path.abspath(chat_file)), "birthdays.csv"),
        os.path.join(os.path.abspath(out_dir), "birthdays.json"),
        os.path.join(os.path.abspath(out_dir), "birthdays.csv"),
        os.path.join(os.getcwd(), "birthdays.json"),
        os.path.join(os.getcwd(), "birthdays.csv"),
    ]
    participant_lookup = {name.casefold(): name for name in participants}
    matched = []
    unmatched = []
    source_path = None

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        source_path = path
        records = []
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                records = [{"name": key, "date": value} for key, value in data.items()]
            elif isinstance(data, list):
                records = data
        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))

        for record in records:
            if not isinstance(record, dict):
                continue
            raw_name = str(record.get("name", "")).strip()
            normalized_date = _normalize_birthday_date(record.get("date"))
            if not raw_name or not normalized_date:
                continue
            canonical_name = participant_lookup.get(raw_name.casefold())
            if canonical_name:
                matched.append({"name": canonical_name, "date": normalized_date, "source": "Manual"})
            else:
                unmatched.append(raw_name)
        break

    matched.sort(key=lambda item: (pd.to_datetime(item["date"], format="%b %d"), item["name"]))
    return matched, unmatched, source_path


def _infer_birthdays_from_chat(df, participants):
    df_messages = df[df["name"] != "System"].copy()
    if df_messages.empty:
        return []

    keywords = (
        "happy birthday",
        "hbd",
        "hbday",
        "bday",
        "birthday",
        "janamdin mubarak",
        "janmdin mubarak",
        "janam din mubarak",
        "janmdin",
    )
    token_map = {}
    for name in participants:
        parts = [part.strip().casefold() for part in re.split(r"\s+", name) if len(part.strip()) >= 4]
        for token in parts + [name.casefold()]:
            token_map.setdefault(token, set()).add(name)

    unique_tokens = {token: next(iter(names)) for token, names in token_map.items() if len(names) == 1}
    evidence = defaultdict(lambda: defaultdict(lambda: {"count": 0, "years": set(), "snippets": []}))

    for _, row in df_messages.iterrows():
        text = str(row.get("message", "")).strip()
        lower_text = text.casefold()
        if not any(keyword in lower_text for keyword in keywords):
            continue

        matches = set()
        for token, participant in unique_tokens.items():
            pattern = r"\b{}\b".format(re.escape(token))
            if re.search(pattern, lower_text):
                matches.add(participant)

        if not matches:
            continue

        date_key = row["date_time"].strftime("%b %d")
        year = int(row["date_time"].year)
        snippet = escape(text[:120] + ("..." if len(text) > 120 else ""))
        for participant in matches:
            bucket = evidence[participant][date_key]
            bucket["count"] += 1
            bucket["years"].add(year)
            if len(bucket["snippets"]) < 2:
                bucket["snippets"].append(snippet)

    inferred = []
    for participant, date_buckets in evidence.items():
        date_str, details = max(
            date_buckets.items(),
            key=lambda item: (item[1]["count"], len(item[1]["years"]), item[0]),
        )
        confidence = "High" if len(details["years"]) >= 2 or details["count"] >= 3 else "Medium"
        inferred.append({
            "name": participant,
            "date": date_str,
            "count": details["count"],
            "years": sorted(details["years"]),
            "confidence": confidence,
            "snippets": details["snippets"],
        })

    inferred.sort(key=lambda item: (pd.to_datetime(item["date"], format="%b %d"), item["name"]))
    return inferred


def _build_overview_cards(group_stats):
    cards = [
        ("Messages", _safe_int(group_stats.get("Total Messages")), "fa-comments"),
        ("Participants", _safe_int(group_stats.get("Unique Users")), "fa-users"),
        ("Media", _safe_int(group_stats.get("Total Media")), "fa-photo-film"),
        ("Links", _safe_int(group_stats.get("Total URLs")), "fa-link"),
        ("Most Active", escape(str(group_stats.get("Most Active Period", "N/A"))), "fa-clock"),
        ("Avg Reply", escape(_format_minutes(group_stats.get("Average Response Time"))), "fa-reply"),
    ]
    return "".join(
        f"""
        <div class="col-md-4 col-xl-2">
            <div class="metric-card h-100">
                <div class="metric-icon"><i class="fas {icon}"></i></div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        </div>
        """
        for label, value, icon in cards
    )


def _build_group_narrative_summary(group_stats, all_user_stats):
    if not all_user_stats:
        return "<p class='mb-0'>Not enough data to generate a group-level narrative summary.</p>"

    total_messages = _safe_int(group_stats.get("Total Messages"))
    total_users = _safe_int(group_stats.get("Unique Users"))
    most_active_period = str(group_stats.get("Most Active Period", "N/A"))
    top_sender, top_sender_stats = max(all_user_stats.items(), key=lambda item: item[1].get("Total Messages", 0))
    top_sender_count = _safe_int(top_sender_stats.get("Total Messages", 0))
    top_sender_share = (top_sender_count / total_messages * 100) if total_messages else 0

    positive = _safe_int(group_stats.get("Positive Messages", 0))
    negative = _safe_int(group_stats.get("Negative Messages", 0))
    if positive > negative * 1.5:
        tone = "generally positive"
    elif negative > max(positive, 1):
        tone = "more critical or tense than positive"
    else:
        tone = "mostly balanced or neutral"

    media = _safe_int(group_stats.get("Total Media", 0))
    links = _safe_int(group_stats.get("Total URLs", 0))
    return (
        f"<p class='mb-0'>This chat spans <strong>{total_users}</strong> participants and <strong>{total_messages}</strong> messages, "
        f"with activity peaking most often at <strong>{escape(most_active_period)}</strong>. "
        f"<strong>{escape(top_sender)}</strong> is the most active contributor with <strong>{top_sender_count}</strong> messages "
        f"({top_sender_share:.0f}% of all visible activity). "
        f"The overall tone is <strong>{tone}</strong>, and the group shared <strong>{media}</strong> media posts and <strong>{links}</strong> links.</p>"
    )


def _build_expanded_group_stats_rows(group_stats, df):
    if df.empty:
        return ""
    active_day = df["date"].value_counts().idxmax()
    active_day_count = int(df["date"].value_counts().max())
    first_message = df["date_time"].min().strftime("%b %d, %Y")
    last_message = df["date_time"].max().strftime("%b %d, %Y")
    rows = [
        ("Total Messages", _safe_int(group_stats.get("Total Messages"))),
        ("Total Participants", _safe_int(group_stats.get("Unique Users"))),
        ("Total Words", _safe_int(group_stats.get("Total Words"))),
        ("Total Emojis", _safe_int(group_stats.get("Total Emojis"))),
        ("Total Media", _safe_int(group_stats.get("Total Media"))),
        ("Total Links", _safe_int(group_stats.get("Total URLs"))),
        ("Total YouTube Links", _safe_int(group_stats.get("Total YouTube URLs"))),
        ("Total Edits", _safe_int(group_stats.get("Total Edits"))),
        ("Total Deletes", _safe_int(group_stats.get("Total Deletes"))),
        ("Positive Messages", _safe_int(group_stats.get("Positive Messages"))),
        ("Negative Messages", _safe_int(group_stats.get("Negative Messages"))),
        ("Average Message Length", f"{_safe_float(group_stats.get('Average Message Length')):.2f}"),
        ("Average Response Time", _format_minutes(group_stats.get("Average Response Time"))),
        ("Most Active Period", escape(str(group_stats.get("Most Active Period", "N/A")))),
        ("Chat Span", f"{first_message} to {last_message}"),
        ("Most Active Day", f"{active_day.strftime('%b %d, %Y')} ({active_day_count} messages)"),
    ]
    return "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)


def _build_period_summary(df):
    if df.empty:
        return "<p class='text-muted'>No period summary available.</p>", ""

    month_counts = df.groupby(df["date_time"].dt.to_period("M")).size().sort_index()
    labels = [str(period) for period in month_counts.index]
    values = [int(value) for value in month_counts.values]
    timeline_chart = render_chartjs({
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Messages",
                "data": values,
                "borderColor": "#0b7285",
                "backgroundColor": "rgba(11, 114, 133, 0.15)",
                "fill": True,
                "tension": 0.15
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False}},
            "scales": {
                "x": {"title": {"display": True, "text": "Month"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Messages"}}
            }
        }
    })

    weekday_counts = df["dayn"].value_counts()
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_summary = ", ".join(f"{day[:3]} {int(weekday_counts.get(day, 0))}" for day in ordered_days)
    month_name_counts = df["monthn"].value_counts()
    top_month_name = month_name_counts.idxmax()
    top_month_hits = int(month_name_counts.max())
    top_year = int(df["year"].value_counts().idxmax())
    top_year_hits = int(df["year"].value_counts().max())
    top_month = month_counts.idxmax()
    top_month_hits_total = int(month_counts.max())

    rows = [
        ("Chat Span", f"{df['date_time'].min().strftime('%b %d, %Y')} to {df['date_time'].max().strftime('%b %d, %Y')}"),
        ("Most Active Year", f"{top_year} ({top_year_hits} messages)"),
        ("Most Active Month", f"{top_month} ({top_month_hits_total} messages)"),
        ("Most Active Calendar Month", f"{top_month_name} ({top_month_hits} messages across all years)"),
        ("Most Active Weekday", f"{df['dayn'].value_counts().idxmax()} ({int(df['dayn'].value_counts().max())} messages)"),
        ("Weekday Split", weekday_summary),
    ]
    return timeline_chart, "".join(f"<tr><th>{label}</th><td>{escape(str(value))}</td></tr>" for label, value in rows)


def _extract_top_keywords(messages, limit=3):
    counter = Counter()
    for message in messages:
        cleaned = clean_message(str(message)).casefold()
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9']{2,}\b", cleaned)
        for word in words:
            if word not in stop_words:
                counter[word] += 1
    return [word for word, _ in counter.most_common(limit)]


def _extract_top_emojis(messages, limit=3):
    counter = Counter()
    for message in messages:
        counter.update(extract_emojis(str(message)))
    return [emoji for emoji, _ in counter.most_common(limit)]


def _build_periodic_topic_analysis(df, periods=8):
    if df.empty:
        return "<tr><td colspan='6' class='text-muted'>No periodic topical analysis available.</td></tr>"

    monthly_groups = []
    for period, period_df in df.groupby(df["date_time"].dt.to_period("M")):
        monthly_groups.append((period, period_df.copy()))

    monthly_groups.sort(key=lambda item: item[0], reverse=True)
    monthly_groups = monthly_groups[:periods]

    rows = []
    for period, period_df in monthly_groups:
        top_user = period_df["name"].value_counts().idxmax()
        top_user_count = int(period_df["name"].value_counts().max())
        top_keywords = _extract_top_keywords(period_df["message"], limit=3)
        top_emojis = _extract_top_emojis(period_df["message"], limit=3)
        top_day = period_df["date"].value_counts().idxmax().strftime("%b %d")

        keyword_html = ", ".join(escape(word) for word in top_keywords) if top_keywords else "No clear keywords"
        emoji_html = " ".join(escape(item) for item in top_emojis) if top_emojis else "None"
        rows.append(
            "<tr>"
            f"<td>{escape(str(period))}</td>"
            f"<td>{len(period_df)}</td>"
            f"<td>{escape(top_user)} ({top_user_count})</td>"
            f"<td>{escape(top_day)}</td>"
            f"<td>{keyword_html}</td>"
            f"<td>{emoji_html}</td>"
            "</tr>"
        )

    return "".join(rows)


def _select_representative_message(messages, max_candidates=40):
    candidates = []
    for message in messages:
        cleaned = clean_message(str(message)).strip()
        if cleaned and cleaned.lower() != "<media omitted>":
            candidates.append(cleaned)
    if not candidates:
        return ""

    candidates = sorted(candidates, key=len, reverse=True)[:max_candidates]
    embeddings = get_sentence_embeddings(candidates)
    if embeddings is not None and len(candidates) > 1:
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return candidates[int(np.argmin(distances))]

    # Fallback when MiniLM is unavailable: use a medium-length message rather than the very longest.
    candidates = sorted(candidates, key=lambda text: abs(len(text) - 90))
    return candidates[0]


def _describe_sentiment(score):
    if score >= 0.15:
        return "mostly positive"
    if score <= -0.15:
        return "more tense or negative"
    return "mostly neutral"


def _build_monthly_summary_paragraphs(df, periods=6):
    if df.empty:
        return "<p class='text-muted'>No monthly summaries available.</p>"

    month_groups = []
    for period, period_df in df.groupby(df["date_time"].dt.to_period("M")):
        month_groups.append((period, period_df.copy()))

    month_groups.sort(key=lambda item: item[0], reverse=True)
    month_groups = month_groups[:periods]

    cards = []
    for period, period_df in month_groups:
        total_messages = len(period_df)
        top_user_counts = period_df["name"].value_counts()
        top_user = top_user_counts.idxmax()
        top_user_count = int(top_user_counts.max())
        share = (top_user_count / total_messages) * 100 if total_messages else 0
        keywords = _extract_top_keywords(period_df["message"], limit=4)
        emojis = _extract_top_emojis(period_df["message"], limit=3)
        media_count = int(period_df.get("mediacount", pd.Series(dtype=int)).sum()) if "mediacount" in period_df.columns else 0
        link_count = int(period_df.get("urlcount", pd.Series(dtype=int)).sum()) if "urlcount" in period_df.columns else 0
        representative = _select_representative_message(period_df["message"])

        sentiment_sample = [
            predict_sentiment(str(message))[0]
            for message in period_df["message"].head(80)
            if str(message).strip()
        ]
        avg_sentiment = float(np.mean(sentiment_sample)) if sentiment_sample else 0.0
        sentiment_label = _describe_sentiment(avg_sentiment)

        busiest_day = period_df["date"].value_counts().idxmax().strftime("%b %d")
        keyword_text = ", ".join(keywords[:3]) if keywords else "no single dominant topic"
        emoji_text = " ".join(emojis) if emojis else "no standout emoji pattern"

        paragraph_parts = [
            f"In {period}, the group exchanged {total_messages} messages and the conversation was {sentiment_label}.",
            f"{top_user} led the month with {top_user_count} messages ({share:.0f}% of the activity), and the busiest day was {busiest_day}.",
        ]

        if keywords:
            paragraph_parts.append(f"Recurring themes included {keyword_text}.")
        else:
            paragraph_parts.append("The discussion did not show strong repeating keywords.")

        if media_count or link_count:
            paragraph_parts.append(f"The month included {media_count} media posts and {link_count} shared links.")

        if emojis:
            paragraph_parts.append(f"Common emoji signals were {emoji_text}.")

        if representative:
            snippet = representative[:180] + ("..." if len(representative) > 180 else "")
            paragraph_parts.append(f"Representative message: \"{escape(snippet)}\"")

        paragraph = " ".join(paragraph_parts)
        cards.append(
            f"""
            <div class="col-lg-6">
                <div class="mini-table-card h-100">
                    <h4>{escape(str(period))}</h4>
                    <p class="mb-0">{paragraph}</p>
                </div>
            </div>
            """
        )

    return "".join(cards)


def _get_month_groups(df):
    groups = []
    if df.empty:
        return groups
    for period, period_df in df.groupby(df["date_time"].dt.to_period("M")):
        groups.append((period, period_df.copy()))
    groups.sort(key=lambda item: item[0])
    return groups


def _build_user_monthly_activity_timeline(df_user):
    groups = _get_month_groups(df_user)
    if not groups:
        return "<p class='text-center text-muted'>No monthly activity available.</p>"
    labels = [str(period) for period, _ in groups]
    values = [len(period_df) for _, period_df in groups]
    return render_chartjs({
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Messages",
                "data": values,
                "backgroundColor": "#25d366",
                "borderColor": "#128c7e",
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False}},
            "scales": {
                "x": {"title": {"display": True, "text": "Month"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Messages"}}
            }
        }
    })


def _build_user_periodic_topic_analysis(df_user, periods=8):
    groups = _get_month_groups(df_user)
    if not groups:
        return "<tr><td colspan='6' class='text-muted'>No periodic topical analysis available.</td></tr>"
    rows = []
    for period, period_df in reversed(groups[-periods:]):
        busiest_day = period_df["date"].value_counts().idxmax().strftime("%b %d")
        top_keywords = _extract_top_keywords(period_df["message"], limit=3)
        top_emojis = _extract_top_emojis(period_df["message"], limit=3)
        media_count = int(period_df["mediacount"].sum()) if "mediacount" in period_df.columns else 0
        link_count = int(period_df["urlcount"].sum()) if "urlcount" in period_df.columns else 0
        rows.append(
            "<tr>"
            f"<td>{escape(str(period))}</td>"
            f"<td>{len(period_df)}</td>"
            f"<td>{escape(busiest_day)}</td>"
            f"<td>{', '.join(escape(word) for word in top_keywords) if top_keywords else 'No clear keywords'}</td>"
            f"<td>{' '.join(escape(emoji) for emoji in top_emojis) if top_emojis else 'None'}</td>"
            f"<td>{media_count} media / {link_count} links</td>"
            "</tr>"
        )
    return "".join(rows)


def _build_user_monthly_summary_paragraphs(df_user, periods=6):
    groups = _get_month_groups(df_user)
    if not groups:
        return "<p class='text-muted'>No monthly summaries available.</p>"

    cards = []
    for period, period_df in reversed(groups[-periods:]):
        total_messages = len(period_df)
        keywords = _extract_top_keywords(period_df["message"], limit=4)
        emojis = _extract_top_emojis(period_df["message"], limit=3)
        media_count = int(period_df["mediacount"].sum()) if "mediacount" in period_df.columns else 0
        link_count = int(period_df["urlcount"].sum()) if "urlcount" in period_df.columns else 0
        busiest_day = period_df["date"].value_counts().idxmax().strftime("%b %d")
        representative = _select_representative_message(period_df["message"])
        sentiment_sample = [
            predict_sentiment(str(message))[0]
            for message in period_df["message"].head(80)
            if str(message).strip()
        ]
        sentiment_label = _describe_sentiment(float(np.mean(sentiment_sample)) if sentiment_sample else 0.0)
        paragraph_parts = [
            f"In {period}, this user posted {total_messages} messages and their tone was {sentiment_label}.",
            f"The busiest day was {busiest_day}.",
        ]
        if keywords:
            paragraph_parts.append(f"Recurring themes included {', '.join(keywords[:3])}.")
        if media_count or link_count:
            paragraph_parts.append(f"They shared {media_count} media posts and {link_count} links.")
        if emojis:
            paragraph_parts.append(f"Common emoji signals were {' '.join(emojis)}.")
        if representative:
            snippet = representative[:180] + ("..." if len(representative) > 180 else "")
            paragraph_parts.append(f"Representative message: \"{escape(snippet)}\"")
        cards.append(
            f"""
            <div class="col-lg-6">
                <div class="mini-table-card h-100">
                    <h4>{escape(str(period))}</h4>
                    <p class="mb-0">{' '.join(paragraph_parts)}</p>
                </div>
            </div>
            """
        )
    return "".join(cards)


def _build_user_participation_share_chart(df_all, username):
    df_non_system = df_all[df_all["name"] != "System"].copy()
    df_user = df_non_system[df_non_system["name"] == username].copy()
    if df_user.empty:
        return "<p class='text-center text-muted'>No participation share data available.</p>"
    total_by_month = df_non_system.groupby(df_non_system["date_time"].dt.to_period("M")).size()
    user_by_month = df_user.groupby(df_user["date_time"].dt.to_period("M")).size()
    periods = sorted(total_by_month.index.union(user_by_month.index))
    labels = [str(period) for period in periods]
    shares = []
    for period in periods:
        total = int(total_by_month.get(period, 0))
        user_total = int(user_by_month.get(period, 0))
        shares.append(round((user_total / total) * 100, 2) if total else 0)
    return render_chartjs({
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Share of Group Messages (%)",
                "data": shares,
                "borderColor": "#ff9f1c",
                "backgroundColor": "rgba(255, 159, 28, 0.18)",
                "fill": True,
                "tension": 0.15
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False}},
            "scales": {
                "x": {"title": {"display": True, "text": "Month"}},
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Message Share %"}}
            }
        }
    })


def _build_user_interaction_trends(df_all, username, periods=8):
    df_non_system = df_all[df_all["name"] != "System"].copy().sort_values("date_time")
    if df_non_system.empty or username not in set(df_non_system["name"]):
        return "<tr><td colspan='4' class='text-muted'>No interaction trend data available.</td></tr>"

    interactions_by_month = defaultdict(Counter)
    prev_row = None
    for _, row in df_non_system.iterrows():
        if prev_row is not None:
            if (row["date_time"] - prev_row["date_time"]) < pd.Timedelta(hours=1):
                if row["name"] == username and prev_row["name"] != username:
                    interactions_by_month[row["date_time"].to_period("M")][prev_row["name"]] += 1
                elif prev_row["name"] == username and row["name"] != username:
                    interactions_by_month[row["date_time"].to_period("M")][row["name"]] += 1
        prev_row = row

    periods_list = sorted(interactions_by_month.keys(), reverse=True)[:periods]
    if not periods_list:
        return "<tr><td colspan='4' class='text-muted'>No interaction trend data available.</td></tr>"

    rows = []
    for period in periods_list:
        if interactions_by_month[period]:
            partner, count = interactions_by_month[period].most_common(1)[0]
            rows.append(f"<tr><td>{escape(str(period))}</td><td>{escape(partner)}</td><td>{count}</td><td>{', '.join(escape(name) for name, _ in interactions_by_month[period].most_common(3))}</td></tr>")
        else:
            rows.append(f"<tr><td>{escape(str(period))}</td><td>None</td><td>0</td><td>No meaningful interactions</td></tr>")
    return "".join(rows)


def _build_user_content_mix_chart(df_user):
    groups = _get_month_groups(df_user)
    if not groups:
        return "<p class='text-center text-muted'>No content mix data available.</p>"
    labels = [str(period) for period, _ in groups]
    media_values = [int(period_df["mediacount"].sum()) if "mediacount" in period_df.columns else 0 for _, period_df in groups]
    link_values = [int(period_df["urlcount"].sum()) if "urlcount" in period_df.columns else 0 for _, period_df in groups]
    text_values = [
        max(len(period_df) - media - links, 0)
        for (_, period_df), media, links in zip(groups, media_values, link_values)
    ]
    return render_chartjs({
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Text Messages", "data": text_values, "backgroundColor": "#4caf50"},
                {"label": "Media", "data": media_values, "backgroundColor": "#42a5f5"},
                {"label": "Links", "data": link_values, "backgroundColor": "#ab47bc"},
            ]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False}},
            "scales": {
                "x": {"stacked": True, "title": {"display": True, "text": "Month"}},
                "y": {"stacked": True, "beginAtZero": True, "title": {"display": True, "text": "Count"}}
            }
        }
    })


def _build_user_overview_cards(user_stats, df_all, username, interaction_matrix):
    df_non_system = df_all[df_all["name"] != "System"].copy()
    user_total = _safe_int(user_stats.get("Total Messages", 0))
    group_total = max(len(df_non_system), 1)
    participation_share = (user_total / group_total) * 100

    user_df = df_non_system[df_non_system["name"] == username].copy()
    if not user_df.empty:
        monthly_counts = user_df.groupby(user_df["date_time"].dt.to_period("M")).size()
        top_month = str(monthly_counts.idxmax())
        top_month_count = int(monthly_counts.max())
    else:
        top_month = "N/A"
        top_month_count = 0

    interactions = interaction_matrix.get(username, {}) if interaction_matrix else {}
    top_partner, top_partner_count = max(interactions.items(), key=lambda item: item[1], default=("None", 0))

    cards = [
        ("Messages", user_total, "fa-comments"),
        ("Share of Group", f"{participation_share:.1f}%", "fa-chart-pie"),
        ("Most Active", escape(str(user_stats.get("Most Active Period", "N/A"))), "fa-clock"),
        ("Avg Reply", escape(_format_minutes(user_stats.get("Average Response Time", 0))), "fa-reply"),
        ("Top Partner", escape(str(top_partner)), "fa-user-group"),
        ("Top Month", f"{escape(top_month)} ({top_month_count})" if top_month != "N/A" else "N/A", "fa-calendar"),
    ]

    return "".join(
        f"""
        <div class="col-md-4 col-xl-2">
            <div class="metric-card h-100">
                <div class="metric-icon"><i class="fas {icon}"></i></div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        </div>
        """
        for label, value, icon in cards
    )


def _build_group_highlights(all_user_stats, user_roles, social_data):
    if not all_user_stats:
        return ""

    top_message_user = max(all_user_stats.items(), key=lambda item: item[1].get("Total Messages", 0))
    top_media_user = max(all_user_stats.items(), key=lambda item: item[1].get("Total Media", 0))
    top_link_user = max(all_user_stats.items(), key=lambda item: item[1].get("Total URLs", 0))
    top_emoji_user = max(all_user_stats.items(), key=lambda item: item[1].get("Total Emojis", 0))
    top_words_user = max(all_user_stats.items(), key=lambda item: item[1].get("Unique Words Count", 0))
    fastest_user = min(
        all_user_stats.items(),
        key=lambda item: item[1].get("Average Response Time", float("inf")) if item[1].get("Average Response Time", 0) > 0 else float("inf")
    )
    top_initiator = max(social_data.get("initiators", {}).items(), key=lambda item: item[1], default=(None, 0))

    highlights = [
        ("Most Active", top_message_user[0], f"{_safe_int(top_message_user[1].get('Total Messages'))} messages"),
        ("Most Expressive", top_emoji_user[0], f"{_safe_int(top_emoji_user[1].get('Total Emojis'))} emojis"),
        ("Top Sharer", top_media_user[0], f"{_safe_int(top_media_user[1].get('Total Media'))} media messages"),
        ("Link Dropper", top_link_user[0], f"{_safe_int(top_link_user[1].get('Total URLs'))} links"),
        ("Vocabulary Lead", top_words_user[0], f"{_safe_int(top_words_user[1].get('Unique Words Count'))} unique words"),
        ("Fastest Replier", fastest_user[0], _format_minutes(fastest_user[1].get("Average Response Time"))),
    ]
    if top_initiator[0]:
        highlights.append(("Conversation Starter", top_initiator[0], f"{_safe_int(top_initiator[1])} conversation starts"))

    role_counts = Counter(role_data.get("role", "Member") for role_data in user_roles.values() if role_data)
    if role_counts:
        top_role, role_count = role_counts.most_common(1)[0]
        highlights.append(("Most Common Role", top_role, f"{role_count} members"))

    return "".join(
        f"""
        <div class="col-md-6 col-xl-3">
            <div class="highlight-card h-100">
                <div class="highlight-kicker">{escape(title)}</div>
                <div class="highlight-name">{escape(name)}</div>
                <div class="highlight-detail">{escape(detail)}</div>
            </div>
        </div>
        """
        for title, name, detail in highlights
    )


def _build_leaderboards(all_user_stats):
    if not all_user_stats:
        return ""

    def top_rows(metric_key, formatter=str, reverse=True):
        sorted_items = sorted(
            all_user_stats.items(),
            key=lambda item: _safe_float(item[1].get(metric_key, 0)),
            reverse=reverse,
        )[:5]
        return [[escape(name), escape(formatter(stats.get(metric_key, 0)))] for name, stats in sorted_items]

    response_rows = [
        (name, stats) for name, stats in all_user_stats.items()
        if _safe_float(stats.get("Average Response Time", 0)) > 0
    ]
    response_rows = sorted(
        response_rows,
        key=lambda item: _safe_float(item[1].get("Average Response Time", 0)),
    )[:5]

    tables = [
        ("Top Message Senders", ["User", "Messages"], top_rows("Total Messages", lambda value: f"{_safe_int(value)}")),
        ("Fastest Responders", ["User", "Avg Reply"], [[escape(name), escape(_format_minutes(stats.get("Average Response Time", 0)))] for name, stats in response_rows]),
        ("Media Sharers", ["User", "Media"], top_rows("Total Media", lambda value: f"{_safe_int(value)}")),
        ("Link Sharers", ["User", "Links"], top_rows("Total URLs", lambda value: f"{_safe_int(value)}")),
        ("Most Positive", ["User", "Positive Msgs"], top_rows("Positive Messages", lambda value: f"{_safe_int(value)}")),
        ("Night Owls", ["User", "Night Msgs"], top_rows("Night Messages", lambda value: f"{_safe_int(value)}")),
    ]
    return "".join(_build_html_table(title, headers, rows) for title, headers, rows in tables)


def _build_roles_table(all_user_stats, user_roles):
    rows = []
    for name in sorted(all_user_stats):
        role_data = user_roles.get(name) or {}
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{escape(role_data.get('role', 'Member'))}</td><td>{escape(str(all_user_stats[name].get('Most Active Period', 'N/A')))}</td><td>{_safe_int(all_user_stats[name].get('Total Messages'))}</td><td>{escape(role_data.get('description', 'A valuable group participant.'))}</td></tr>"
        )
    return "".join(rows)


def _build_interaction_summary(social_data):
    initiators = social_data.get("initiators", {})
    interaction_matrix = social_data.get("interaction_matrix", {})

    initiator_rows = [[escape(name), str(_safe_int(count))] for name, count in sorted(initiators.items(), key=lambda item: item[1], reverse=True)[:8]]

    pair_counts = []
    for source, targets in interaction_matrix.items():
        for target, count in targets.items():
            if count > 0:
                pair_counts.append((source, target, count))
    pair_counts.sort(key=lambda item: item[2], reverse=True)
    pair_rows = [[escape(source), escape(target), str(_safe_int(count))] for source, target, count in pair_counts[:8]]

    return (
        _build_html_table("Conversation Starters", ["User", "Starts"], initiator_rows)
        + _build_html_table("Strongest Interaction Pairs", ["From", "To", "Interactions"], pair_rows)
    )


def _build_topics_summary(group_stats):
    emoji_html = " ".join(
        f"<span class='topic-chip'>{escape(emoji)} ({count})</span>"
        for emoji, count in group_stats.get("Top 5 Emojis", [])
    ) or "<p class='text-muted mb-0'>No emoji summary available.</p>"
    return {
        "unigrams": group_stats.get("Common Unigrams", "") or "<li>No common unigram data available.</li>",
        "bigrams": group_stats.get("Common Bigrams", "") or "<li>No common bigram data available.</li>",
        "trigrams": group_stats.get("Common Trigrams", "") or "<li>No common trigram data available.</li>",
        "emojis": emoji_html,
    }


def _build_birthday_rows(manual_birthdays, inferred_birthdays):
    manual_rows = "".join(
        f"<tr><td>{escape(item['name'])}</td><td>{escape(item['date'])}</td><td>{escape(item['source'])}</td></tr>"
        for item in manual_birthdays
    ) or "<tr><td colspan='3' class='text-muted'>No manual birthday file found.</td></tr>"

    inferred_rows = ""
    for item in inferred_birthdays:
        evidence = "; ".join(item["snippets"]) if item["snippets"] else "Birthday-style wishes detected."
        inferred_rows += (
            f"<tr><td>{escape(item['name'])}</td><td>{escape(item['date'])}</td><td>{escape(item['confidence'])}</td>"
            f"<td>{_safe_int(item['count'])} wish messages / years: {', '.join(str(year) for year in item['years'])}</td>"
            f"<td>{evidence}</td></tr>"
        )
    if not inferred_rows:
        inferred_rows = "<tr><td colspan='5' class='text-muted'>No probable birthdays inferred from chat text.</td></tr>"

    return manual_rows, inferred_rows

class WhatsAppAnalyzer:
    def __init__(self, chat_file, out_dir="."):
        self.chat_file = chat_file
        self.out_dir = out_dir
        self.parser = Parser(self.chat_file)
        self.df = df_basic_cleanup(self.parser.parse_chat_data())

    def generate_report(self, users=None):
        """
        Generates HTML reports for specified users and a group summary.
        """
        all_users = self.df["name"].unique()
        if users is None:
            # Exclude System from default user reports
            users = [u for u in all_users if u != "System"]

        os.makedirs(self.out_dir, exist_ok=True)
        shared_user_relationship_graph = plot_user_relationship_graph(self.df)

        # 1. Collect basic stats for all users first to compute group-wide metrics
        all_user_stats = {}
        for name in users:
            print(f"Collecting stats for {name}...")
            all_user_stats[name] = basic_stats(
                self.df,
                name,
                shared_user_relationship_graph=shared_user_relationship_graph,
            )

        # 2. Compute group-wide psychological and social metrics
        from .analysis_utils import calculate_ocean_traits, analyze_social_dynamics, calculate_group_roles, plot_conflict_heatmap, calculate_temporal_ocean
        from .plot_utils import plot_personality_radar, plot_interaction_matrix
        
        ocean_percentiles = calculate_ocean_traits(all_user_stats)
        user_roles = calculate_group_roles(all_user_stats, ocean_percentiles)
        # 2.5 Compute Group Averages for Benchmarking
        group_avg = {}
        numeric_keys = ['Total Messages', 'Total Words', 'Total Emojis', 'Total Media', 'Average Message Length', 'Average Response Time']
        for key in numeric_keys:
            vals = [s.get(key, 0) for s in all_user_stats.values()]
            group_avg[key] = sum(vals) / len(vals) if vals else 0

        social_data = analyze_social_dynamics(self.df)
        interaction_matrix = social_data.get('interaction_matrix', {})

        user_reports = []

        
        # 3. Finalize each user report with individual and comparative data
        for name, user_stats in all_user_stats.items():
            user_df = self.df[(self.df["name"] == name)].copy()
            # Benchmarking strings
            user_stats['avg_msgs'] = group_avg['Total Messages']
            user_stats['avg_len'] = group_avg['Average Message Length']
            user_stats['avg_speed'] = group_avg['Average Response Time']
            
            user_stats['msg_performance'] = "Above" if user_stats['Total Messages'] > group_avg['Total Messages'] else "Below"
            user_stats['msg_badge_class'] = "bg-success" if user_stats['Total Messages'] > group_avg['Total Messages'] else "bg-secondary"
            
            user_stats['len_performance'] = "More Detailed" if user_stats['Average Message Length'] > group_avg['Average Message Length'] else "More Concise"
            user_stats['len_badge_class'] = "bg-success" if user_stats['Average Message Length'] > group_avg['Average Message Length'] else "bg-secondary"
            
            user_stats['speed_performance'] = "Faster" if user_stats['Average Response Time'] < group_avg['Average Response Time'] else "Slower"
            user_stats['speed_badge_class'] = "bg-success" if user_stats['Average Response Time'] < group_avg['Average Response Time'] else "bg-secondary"

            # Add personality radar
            user_ocean = ocean_percentiles.get(name, {})
            user_stats['Personality Radar Chart'] = plot_personality_radar(user_ocean, name)
            
            # Add Phase 4 & 5 metrics
            user_stats['Conflict Heatmap'] = plot_conflict_heatmap(self.df, name)
            user_stats['Personality Evolution'] = calculate_temporal_ocean(self.df, name)
            user_stats['Monthly Activity Timeline'] = _build_user_monthly_activity_timeline(user_df)
            user_stats['Periodic Topical Analysis Rows'] = _build_user_periodic_topic_analysis(user_df)
            user_stats['Monthly Summary Cards'] = _build_user_monthly_summary_paragraphs(user_df)
            user_stats['Participation Share Over Time'] = _build_user_participation_share_chart(self.df, name)
            user_stats['Interaction Trends Rows'] = _build_user_interaction_trends(self.df, name)
            user_stats['Content Mix Over Time'] = _build_user_content_mix_chart(user_df)
            
            # Add interaction radar/bar
            user_stats['Top Interactions'] = plot_interaction_matrix(interaction_matrix, name)
            
            # Add role data
            role_data = user_roles.get(name, {})
            user_stats['Assigned Role'] = role_data.get('role', 'Member')
            user_stats['Role Description'] = role_data.get('description', 'A valuable group participant.')
            user_stats['Role Icon'] = role_data.get('icon', 'fa-user')
            user_stats['Overview Cards'] = _build_user_overview_cards(user_stats, self.df, name, interaction_matrix)

            # Generate HTML for top 5 emojis
            top_5_emojis_html_str = " ".join(
                [f"{escape(emoji_val)} ({count})" for emoji_val, count in user_stats["Top 5 Emojis"]]
            )
            user_stats['top_5_emojis_html'] = top_5_emojis_html_str

            # Build initials
            initials = "".join(part[0].upper() for part in name.split() if part)[:2]

            # Clear temporary data not used in formatting the template
            keys_to_remove = ["Top 5 Emojis", "Behavioral Traits", "abuse_raw_count"]
            user_stats_clean = {k: v for k, v in user_stats.items() if k not in keys_to_remove}

            safe_name = escape(name)
            try:
                final_html = html_template.format(
                    name=safe_name, 
                    initials=escape(initials), 
                    group_avg=group_avg, **user_stats_clean
                )
            except KeyError as e:
                print(f"Warning: Missing key {e} in user stats for {name}. Using empty string fallback.")
                clean_key = str(e).strip("'")
                user_stats_clean[clean_key] = "N/A"
                final_html = html_template.format(
                    name=safe_name, initials=escape(initials), group_avg=group_avg, **user_stats_clean
                )

            # Sanitize filename
            safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "user"
            filename = f"{safe_filename}_report.html"
            output_path = os.path.join(self.out_dir, filename)

            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(final_html)
            
            user_reports.append({"name": name, "filename": filename})
            print(f"Report for {name} has been generated and saved at {output_path}")

        # Generate Group Summary (Index)
        self.generate_group_summary(
            user_reports,
            shared_user_relationship_graph,
            all_user_stats,
            user_roles,
            social_data,
        )

    def generate_group_summary(self, user_reports, shared_user_relationship_graph, all_user_stats, user_roles, social_data):
        """Generates an index.html with group-level statistics and links to user reports."""
        print("Generating group summary...")
        group_stats = basic_stats(
            self.df, 
            username=None, 
            shared_user_relationship_graph=shared_user_relationship_graph
        )
        participants = sorted(all_user_stats.keys())
        group_df = self.df[self.df["name"] != "System"].copy()
        timeline_chart, period_summary_rows = _build_period_summary(group_df)
        periodic_topic_rows = _build_periodic_topic_analysis(group_df)
        monthly_summary_cards = _build_monthly_summary_paragraphs(group_df)
        overview_cards = _build_overview_cards(group_stats)
        expanded_group_stats_rows = _build_expanded_group_stats_rows(group_stats, group_df)
        group_highlights = _build_group_highlights(all_user_stats, user_roles, social_data)
        leaderboard_tables = _build_leaderboards(all_user_stats)
        role_rows = _build_roles_table(all_user_stats, user_roles)
        interaction_summary = _build_interaction_summary(social_data)
        topic_sections = _build_topics_summary(group_stats)
        group_narrative_summary = _build_group_narrative_summary(group_stats, all_user_stats)
        manual_birthdays, unmatched_birthdays, manual_birthday_path = _load_manual_birthdays(self.chat_file, self.out_dir, participants)
        inferred_birthdays = _infer_birthdays_from_chat(self.df, participants)
        manual_birthday_rows, inferred_birthday_rows = _build_birthday_rows(manual_birthdays, inferred_birthdays)

        if manual_birthday_path:
            manual_birthday_note = f"Loaded manual birthdays from {escape(os.path.relpath(manual_birthday_path, os.getcwd()))}."
            if unmatched_birthdays:
                manual_birthday_note += f" Unmatched names skipped: {escape(', '.join(sorted(set(unmatched_birthdays))))}."
        else:
            manual_birthday_note = "No manual birthday file found. Supported files: birthdays.json or birthdays.csv near the chat file, output directory, or project root."
        inferred_birthday_note = "Probable birthdays are inferred from birthday-style wish messages in the chat and may be incomplete or incorrect."

        user_links_html = ""
        for user in sorted(user_reports, key=lambda x: x['name']):
            role_label = (user_roles.get(user['name']) or {}).get('role', 'Member')
            total_messages = _safe_int(all_user_stats.get(user['name'], {}).get('Total Messages'))
            active_period = escape(str(all_user_stats.get(user['name'], {}).get('Most Active Period', 'N/A')))
            user_links_html += f"""
                <div class="col-md-4">
                    <div class="user-card">
                        <a class="user-link flex-column" href="{user['filename']}">
                            <span class="participant-name">{escape(user['name'])}</span>
                            <span class="participant-meta">{escape(role_label)} | {total_messages} msgs | {active_period}</span>
                        </a>
                    </div>
                </div>
            """

        # Format key removal
        keys_to_remove = ["Top 5 Emojis", "Behavioral Traits", "abuse_raw_count"]
        group_stats_clean = {k: v for k, v in group_stats.items() if k not in keys_to_remove}

        final_index_html = index_template.format(
            user_links=user_links_html,
            overview_cards=overview_cards,
            group_narrative_summary=group_narrative_summary,
            expanded_group_stats_rows=expanded_group_stats_rows,
            group_activity_timeline=timeline_chart,
            period_summary_rows=period_summary_rows,
            periodic_topic_rows=periodic_topic_rows,
            monthly_summary_cards=monthly_summary_cards,
            group_highlights=group_highlights,
            leaderboard_tables=leaderboard_tables,
            role_rows=role_rows,
            interaction_summary=interaction_summary,
            common_unigrams=topic_sections["unigrams"],
            common_bigrams=topic_sections["bigrams"],
            common_trigrams=topic_sections["trigrams"],
            top_group_emojis=topic_sections["emojis"],
            manual_birthday_note=manual_birthday_note,
            manual_birthday_rows=manual_birthday_rows,
            inferred_birthday_note=inferred_birthday_note,
            inferred_birthday_rows=inferred_birthday_rows,
            **group_stats_clean
        )

        index_path = os.path.join(self.out_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_index_html)
        
        print(f"Group summary generated at {index_path}")
