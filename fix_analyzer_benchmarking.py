import re

with open('whatsapp_analyzer/analyzer.py', 'r') as f:
    content = f.read()

# Pre-calculate benchmarking badges
benchmark_logic = '''
        # 2.5 Compute Group Averages for Benchmarking
        group_avg = {}
        numeric_keys = ['Total Messages', 'Total Words', 'Total Emojis', 'Total Media', 'Average Message Length', 'Average Response Time']
        for key in numeric_keys:
            vals = [s.get(key, 0) for s in all_user_stats.values()]
            group_avg[key] = sum(vals) / len(vals) if vals else 0
'''

# We need to replace the format call to include pre-calculated strings for the badges
# Or better, just simplify the template to not use logic

with open('whatsapp_analyzer/constants.py', 'r') as f:
    html_template = f.read()

# Replace logic in template with simple placeholders
html_template = html_template.replace("{ 'bg-success' if Total Messages > group_avg[Total Messages] else 'bg-secondary' }", "{msg_badge_class}")
html_template = html_template.replace("{ 'Above' if Total Messages > group_avg[Total Messages] else 'Below' }", "{msg_performance}")
html_template = html_template.replace("{ 'bg-success' if Average Message Length > group_avg[Average Message Length] else 'bg-secondary' }", "{len_badge_class}")
html_template = html_template.replace("{ 'More Detailed' if Average Message Length > group_avg[Average Message Length] else 'More Concise' }", "{len_performance}")
html_template = html_template.replace("{ 'bg-success' if Average Response Time < group_avg[Average Response Time] else 'bg-secondary' }", "{speed_badge_class}")
html_template = html_template.replace("{ 'Faster' if Average Response Time < group_avg[Average Response Time] else 'Slower' }", "{speed_performance}")

# Fix keys in benchmarking table
html_template = html_template.replace("{group_avg[Total Messages]:.1f}", "{avg_msgs:.1f}")
html_template = html_template.replace("{group_avg[Average Message Length]:.1f}", "{avg_len:.1f}")
html_template = html_template.replace("{group_avg[Average Response Time]:.1f}", "{avg_speed:.1f}")

with open('whatsapp_analyzer/constants.py', 'w') as f:
    f.write(html_template)

# Now update analyzer.py to provide these placeholders
new_loop_logic = '''
        # 3. Finalize each user report with individual and comparative data
        for name, user_stats in all_user_stats.items():
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
'''

content = re.sub(r'# 3\. Finalize each user report with individual and comparative data\s+for name, user_stats in all_user_stats\.items\(\):', 
                 new_loop_logic, content)

with open('whatsapp_analyzer/analyzer.py', 'w') as f:
    f.write(content)
