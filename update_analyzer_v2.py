import re

with open('whatsapp_analyzer/analyzer.py', 'r') as f:
    content = f.read()

# 1. Update generate_report to include Conflict Heatmap and Personality Evolution
content = content.replace('from .analysis_utils import calculate_ocean_traits, analyze_social_dynamics, calculate_group_roles', 
                          'from .analysis_utils import calculate_ocean_traits, analyze_social_dynamics, calculate_group_roles, plot_conflict_heatmap, calculate_temporal_ocean')

# In generate_report loop:
content = content.replace("user_stats['Personality Radar Chart'] = plot_personality_radar(user_ocean, name)",
                          '''user_stats['Personality Radar Chart'] = plot_personality_radar(user_ocean, name)
            
            # Add Phase 4 & 5 metrics
            user_stats['Conflict Heatmap'] = plot_conflict_heatmap(self.df, name)
            user_stats['Personality Evolution'] = calculate_temporal_ocean(self.df, name)''')

# 2. Add Benchmarking logic
benchmark_logic = '''
        # 2.5 Compute Group Averages for Benchmarking
        group_avg = {}
        numeric_keys = ['Total Messages', 'Total Words', 'Total Emojis', 'Total Media', 'Average Message Length', 'Average Response Time']
        for key in numeric_keys:
            vals = [s.get(key, 0) for s in all_user_stats.values()]
            group_avg[key] = sum(vals) / len(vals) if vals else 0
'''

content = content.replace('user_roles = calculate_group_roles(all_user_stats, ocean_percentiles)',
                          'user_roles = calculate_group_roles(all_user_stats, ocean_percentiles)' + benchmark_logic)

# Add group_avg to formatting
content = content.replace('**user_stats_clean', 'group_avg=group_avg, **user_stats_clean')

with open('whatsapp_analyzer/analyzer.py', 'w') as f:
    f.write(content)
