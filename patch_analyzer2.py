import re

with open('whatsapp_analyzer/analyzer.py', 'r') as f:
    content = f.read()

# Add social dynamics calculation to generate_report
new_gen_report_social = '''    def generate_report(self, users=None):
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
        from .analysis_utils import calculate_ocean_traits, analyze_social_dynamics
        from .plot_utils import plot_personality_radar, plot_interaction_matrix
        
        ocean_percentiles = calculate_ocean_traits(all_user_stats)
        social_data = analyze_social_dynamics(self.df)
        interaction_matrix = social_data['interaction_matrix']

        user_reports = []

        # 3. Finalize each user report with individual and comparative data
        for name, user_stats in all_user_stats.items():
            # Add personality radar
            user_ocean = ocean_percentiles.get(name, {})
            user_stats['Personality Radar Chart'] = plot_personality_radar(user_ocean, name)
            
            # Add interaction radar/bar
            user_stats['Top Interactions'] = plot_interaction_matrix(interaction_matrix, name)

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
                    **user_stats_clean
                )
            except KeyError as e:
                print(f"Warning: Missing key {e} in user stats for {name}. Using empty string fallback.")
                # Simple fallback if template expects a key we missed
                clean_key = str(e).strip("'")
                user_stats_clean[clean_key] = "N/A"
                final_html = html_template.format(
                    name=safe_name, initials=escape(initials), **user_stats_clean
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
        self.generate_group_summary(user_reports, shared_user_relationship_graph)'''

content = re.sub(r'def generate_report\(.*?\):.*?(?=\n\n?def |\Z)', new_gen_report_social, content, flags=re.DOTALL)

with open('whatsapp_analyzer/analyzer.py', 'w') as f:
    f.write(content)
