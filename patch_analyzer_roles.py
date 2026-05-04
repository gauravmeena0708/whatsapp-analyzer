import re

with open('whatsapp_analyzer/analyzer.py', 'r') as f:
    content = f.read()

# Update generate_report to compute and include roles
content = content.replace('from .analysis_utils import calculate_ocean_traits, analyze_social_dynamics', 
                          'from .analysis_utils import calculate_ocean_traits, analyze_social_dynamics, calculate_group_roles')

# In generate_report:
content = content.replace('ocean_percentiles = calculate_ocean_traits(all_user_stats)',
                          '''ocean_percentiles = calculate_ocean_traits(all_user_stats)
        user_roles = calculate_group_roles(all_user_stats, ocean_percentiles)''')

content = content.replace("user_stats['Top Interactions'] = plot_interaction_matrix(interaction_matrix, name)",
                          '''user_stats['Top Interactions'] = plot_interaction_matrix(interaction_matrix, name)
            
            # Add role data
            role_data = user_roles.get(name, {})
            user_stats['Assigned Role'] = role_data.get('role', 'Member')
            user_stats['Role Description'] = role_data.get('description', 'A valuable group participant.')
            user_stats['Role Icon'] = role_data.get('icon', 'fa-user')''')

with open('whatsapp_analyzer/analyzer.py', 'w') as f:
    f.write(content)
