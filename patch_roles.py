import re

# 1. Update analysis_utils.py
with open('whatsapp_analyzer/analysis_utils.py', 'r') as f:
    analysis_code = f.read()

role_logic = '''
def calculate_group_roles(all_user_stats, ocean_percentiles):
    """Assign a role to each user based on their personality and stats."""
    from .constants import group_roles
    import numpy as np
    
    user_roles = {}
    users = list(all_user_stats.keys())
    if not users: return {}

    for user in users:
        stats = all_user_stats[user]
        ocean = ocean_percentiles.get(user, {})
        total_msgs = stats.get('Total Messages', 0)
        
        if total_msgs == 0:
            user_roles[user] = None
            continue

        role_scores = {}
        
        # Night Owl (Objective metric)
        night_ratio = stats.get('Night Messages', 0) / total_msgs
        role_scores['The Night Owl'] = night_ratio * 1.5

        # Energizer: Extraversion + Emojis
        emoji_density = stats.get('Total Emojis', 0) / total_msgs
        role_scores['The Energizer'] = (ocean.get('Extraversion', 0.5) + min(emoji_density, 1.0)) / 2

        # Mediator: Agreeableness + Sentiment
        role_scores['The Mediator'] = (ocean.get('Agreeableness', 0.5) + (stats.get('Positive Messages', 0) / total_msgs)) / 2

        # Expert: Conscientiousness + Technical Skills
        skills = stats.get('Behavioral Traits', {}).get('skills', {})
        tech_score = skills.get('technical', 0) / (total_msgs / 10 + 1) # Normalized
        role_scores['The Expert'] = (ocean.get('Conscientiousness', 0.5) + min(tech_score, 1.0)) / 2

        # Social Butterfly: Extraversion + Multi-user interactions
        # (Interaction Partner count could be added, but for now just Extraversion)
        role_scores['The Social Butterfly'] = ocean.get('Extraversion', 0.5) * 1.1

        # Lurker: Inverse of message frequency (relative to group)
        avg_msgs = np.mean([all_user_stats[u].get('Total Messages', 0) for u in users])
        if total_msgs < (avg_msgs * 0.5):
            role_scores['The Lurker'] = (1.0 - (total_msgs / avg_msgs))
        else:
            role_scores['The Lurker'] = 0.0

        # Pick the role with the highest score
        best_role = max(role_scores, key=role_scores.get)
        user_roles[user] = {
            'role': best_role,
            'description': group_roles[best_role]['description'],
            'icon': group_roles[best_role]['icon']
        }

    return user_roles

def calculate_ocean_traits(all_user_stats):'''

analysis_code = analysis_code.replace('def calculate_ocean_traits(all_user_stats):', role_logic + '\ndef calculate_ocean_traits(all_user_stats):')

with open('whatsapp_analyzer/analysis_utils.py', 'w') as f:
    f.write(analysis_code)
