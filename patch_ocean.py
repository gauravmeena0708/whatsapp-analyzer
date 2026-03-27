import re

# Update plot_utils.py
with open('whatsapp_analyzer/plot_utils.py', 'r') as f:
    plot_code = f.read()

new_radar = '''def plot_personality_radar(ocean_scores, username=None):
    if not ocean_scores:
        return "<p class='text-center text-muted' style='margin-top: 50px;'>Personality data unavailable (need more messages).</p>"

    traits = list(ocean_scores.keys())
    values = list(ocean_scores.values())

    title = f'Big Five Personality Traits {"for " + username if username else ""}'
    config = {
        "type": "radar",
        "data": {
            "labels": traits,
            "datasets": [{
                "label": "Percentile",
                "data": values,
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "borderColor": "rgb(255, 99, 132)",
                "pointBackgroundColor": "rgb(255, 99, 132)",
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": "rgb(255, 99, 132)"
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "r": {"angleLines": {"display": True}, "suggestedMin": 0, "suggestedMax": 1, "ticks": {"display": False}}
            }
        }
    }
    return render_chartjs(config)'''

if 'def plot_personality_radar' not in plot_code:
    plot_code += "\n\n" + new_radar

with open('whatsapp_analyzer/plot_utils.py', 'w') as f:
    f.write(plot_code)


# Update analysis_utils.py
with open('whatsapp_analyzer/analysis_utils.py', 'r') as f:
    analysis_code = f.read()

ocean_logic = '''
def calculate_ocean_traits(all_user_stats):
    """
    Calculate OCEAN percentiles for each user based on group-wide stats.
    """
    from .constants import personality_weights
    import numpy as np

    ocean_scores = {}
    users = list(all_user_stats.keys())
    if not users: return {}
    
    # Pre-compute raw scores for each trait and user
    raw_ocean = {user: {trait: 0.0 for trait in personality_weights} for user in users}

    for user in users:
        stats = all_user_stats[user]
        traits = stats.get('Behavioral Traits', {})
        total_msgs = stats.get('Total Messages', 0)
        
        # Don't profile if very low activity
        if total_msgs < 10:
            for trait in personality_weights: raw_ocean[user][trait] = None
            continue
        
        # Map stats to the weights
        data_map = {
            'lexical_diversity': traits.get('lexical_diversity', 0),
            'unique_words_count': stats.get('Unique Words Count', 0),
            'avg_message_length': stats.get('Average Message Length', 0),
            'avg_sentence_length': traits.get('avg_sentence_length', 0),
            'total_words': stats.get('Total Words', 0),
            'total_messages': total_msgs,
            'emoji_density': stats.get('Total Emojis', 0) / total_msgs,
            'exclamation_ratio': traits.get('num_exclamations', 0) / total_msgs,
            'avg_sentiment_polarity': traits.get('avg_sentiment_polarity', 0),
            'abuse_ratio': stats.get('abuse_raw_count', 0) / total_msgs,
            'first_person_ratio': traits.get('first_person_pronouns', 0) / total_msgs
        }

        for trait, weights in personality_weights.items():
            for metric, weight in weights.items():
                raw_ocean[user][trait] += data_map.get(metric, 0) * weight

    # Normalize to percentiles (0 to 1)
    final_ocean = {user: {} for user in users}
    for trait in personality_weights:
        # Only consider users who actually have a score
        trait_scores = [raw_ocean[u][trait] for u in users if raw_ocean[u][trait] is not None]
        if not trait_scores: continue
        
        min_s, max_s = min(trait_scores), max(trait_scores)
        range_s = max_s - min_s
        
        for user in users:
            if raw_ocean[user][trait] is None:
                final_ocean[user][trait] = 0.5
            elif range_s > 0:
                final_ocean[user][trait] = (raw_ocean[user][trait] - min_s) / range_s
            else:
                final_ocean[user][trait] = 0.5

    return final_ocean
'''

if 'def calculate_ocean_traits' not in analysis_code:
    analysis_code = analysis_code.replace('from .plot_utils import (', ocean_logic + '\nfrom .plot_utils import (')

# Update basic_stats to return abuse_raw_count
analysis_code = analysis_code.replace("'Hindi Abuse Counts': abuse_counts_html,", 
                                      "'Hindi Abuse Counts': abuse_counts_html, 'abuse_raw_count': sum(abuse_counts.values()),")

# Add plot_personality_radar to imports in analysis_utils
analysis_code = analysis_code.replace('    plot_skills_radar_chart\n)', '    plot_skills_radar_chart,\n    plot_personality_radar\n)')

with open('whatsapp_analyzer/analysis_utils.py', 'w') as f:
    f.write(analysis_code)
