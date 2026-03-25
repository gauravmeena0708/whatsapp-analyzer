# analyzer.py (inside whatsapp_analyzer)
import os
import re
from html import escape
from .parser import Parser
from .utils import (
    df_basic_cleanup,
)
from .plot_utils import plot_user_relationship_graph
from .analysis_utils import basic_stats
from .constants import html_template, index_template # Used directly in generate_report

class WhatsAppAnalyzer:
    def __init__(self, chat_file, out_dir="."):
        self.chat_file = chat_file
        self.out_dir = out_dir
        self.parser = Parser(self.chat_file)
        self.df = df_basic_cleanup(self.parser.parse_chat_data())

    def generate_report(self, users=None):
        """
        Generates HTML reports for specified users and a group summary.

        Args:
            users (list, optional): A list of usernames for which to generate reports. 
                                   If None, reports are generated for all users. Defaults to None.
        """
        all_users = self.df["name"].unique()
        if users is None:
            users = all_users

        os.makedirs(self.out_dir, exist_ok=True)
        shared_user_relationship_graph = plot_user_relationship_graph(self.df)

        user_reports = []

        for name in users:
            if name != "System":
                # Call basic_stats from analysis_utils
                user_stats = basic_stats(
                    self.df,
                    name,
                    shared_user_relationship_graph=shared_user_relationship_graph,
                )

                # Generate HTML for top 5 emojis and add to user_stats
                top_5_emojis_html_str = " ".join(
                    [f"{escape(emoji_val)} ({count})" for emoji_val, count in user_stats["Top 5 Emojis"]]
                )
                user_stats['top_5_emojis_html'] = top_5_emojis_html_str

                # Build initials for the profile avatar (up to 2 chars)
                initials = "".join(part[0].upper() for part in name.split() if part)[:2]

                # Remove keys not used directly in the template to avoid format() errors
                user_stats.pop("Top 5 Emojis", None)
                user_stats.pop("Behavioral Traits", None)

                safe_name = escape(name)
                final_html = html_template.format(name=safe_name, initials=escape(initials), **user_stats)

                # Sanitize filename
                safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "user"
                filename = f"{safe_filename}_report.html"
                output_path = os.path.join(self.out_dir, filename)

                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(final_html)
                
                user_reports.append({"name": name, "filename": filename})
                print(f"Report for {name} has been generated and saved at {output_path}")

        # Generate Group Summary (Index)
        self.generate_group_summary(user_reports, shared_user_relationship_graph)

    def generate_group_summary(self, user_reports, shared_user_relationship_graph):
        """Generates an index.html with group-level statistics and links to user reports."""
        print("Generating group summary...")
        group_stats = basic_stats(
            self.df, 
            username=None, 
            shared_user_relationship_graph=shared_user_relationship_graph
        )

        user_links_html = ""
        for user in sorted(user_reports, key=lambda x: x['name']):
            user_links_html += f"""
                <div class="col-md-4">
                    <div class="user-card">
                        <a class="user-link" href="{user['filename']}">{escape(user['name'])}</a>
                    </div>
                </div>
            """

        final_index_html = index_template.format(
            user_links=user_links_html,
            **group_stats
        )

        index_path = os.path.join(self.out_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_index_html)
        
        print(f"Group summary generated at {index_path}")
