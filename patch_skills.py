import re

# 1. Update plot_utils.py
with open('whatsapp_analyzer/plot_utils.py', 'r') as f:
    plot_code = f.read()

# Replace plot_skills_radar_chart to accept skill_counts instead of calculating them
new_radar = '''def plot_skills_radar_chart(skill_counts, username=None):
    skills = list(skill_counts.keys())
    counts = list(skill_counts.values())
    if counts and max(counts) > 0:
        max_val = max(counts)
        normalized_counts = [c / max_val for c in counts]
    else:
        return "<p class='text-center text-muted' style='margin-top: 50px;'>No skill categories available.</p>"

    title = f'Skills Radar Chart {"for " + username if username else ""}'
    config = {
        "type": "radar",
        "data": {
            "labels": skills,
            "datasets": [{
                "label": "Skill Level",
                "data": normalized_counts,
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgb(54, 162, 235)",
                "pointBackgroundColor": "rgb(54, 162, 235)",
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": "rgb(54, 162, 235)"
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": False, "text": title}},
            "scales": {
                "r": {"angleLines": {"display": True}, "suggestedMin": 0, "suggestedMax": 1}
            }
        }
    }
    return render_chartjs(config)'''

plot_code = re.sub(r'def plot_skills_radar_chart\(.*?\):.*?(?=\n\n?def |\Z)', new_radar, plot_code, flags=re.DOTALL)

with open('whatsapp_analyzer/plot_utils.py', 'w') as f:
    f.write(plot_code)


# 2. Update analysis_utils.py
with open('whatsapp_analyzer/analysis_utils.py', 'r') as f:
    analysis_code = f.read()

get_skill_scores_code = '''
# Cache embeddings globally
_SKILL_EMBEDDINGS = None

def get_skill_scores(df_filtered):
    """Calculate skill scores based on semantic similarity or fallback to keyword matching."""
    global _SKILL_EMBEDDINGS
    import numpy as np
    from .ml_models import get_sentence_model
    
    model = get_sentence_model()
    
    if model is None:
        skill_counts = {}
        for skill, keywords in skill_keywords.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            skill_counts[skill] = int(df_filtered['clean_message_lower'].str.count(pattern).sum())
        return skill_counts

    if _SKILL_EMBEDDINGS is None:
        _SKILL_EMBEDDINGS = {}
        for skill, keywords in skill_keywords.items():
            text = f"{skill} " + " ".join(keywords[:15])
            emb = model.encode([text], show_progress_bar=False)[0]
            _SKILL_EMBEDDINGS[skill] = emb
            
    msgs = df_filtered['clean_message'].dropna()
    msgs = msgs[msgs.str.len() > 10]
    if len(msgs) > 500:
        msgs = msgs.sample(500, random_state=42)
        
    if msgs.empty:
        return {skill: 0 for skill in skill_keywords.keys()}
        
    msg_embs = model.encode(msgs.tolist(), show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity
    skill_names = list(_SKILL_EMBEDDINGS.keys())
    skill_matrix = np.array([_SKILL_EMBEDDINGS[s] for s in skill_names])
    
    scores = cosine_similarity(msg_embs, skill_matrix)
    
    threshold = 0.25
    aggregated_scores = {}
    for i, skill in enumerate(skill_names):
        matches = scores[:, i]
        valid_matches = matches[matches > threshold]
        aggregated_scores[skill] = int(np.sum(valid_matches) * 5)
        
    return aggregated_scores

def analyze_behavioral_traits(df, username=None):'''

analysis_code = analysis_code.replace('def analyze_behavioral_traits(df, username=None):', get_skill_scores_code)

old_skill_calc = '''    # --- Skill Analysis (Keyword-based) ---
    # Escape each keyword to avoid regex special characters causing silent mismatches
    traits['skills'] = {}
    for skill, keywords in skill_keywords.items():
        pattern = '|'.join(re.escape(kw) for kw in keywords)
        traits['skills'][skill] = int(df_filtered_behavior['clean_message_lower'].str.count(pattern).sum())'''

new_skill_calc = '''    # --- Skill Analysis (Semantic + Keyword Fallback) ---
    traits['skills'] = get_skill_scores(df_filtered_behavior)'''

analysis_code = analysis_code.replace(old_skill_calc, new_skill_calc)

# Update basic_stats to pass skill_counts to plot_skills_radar_chart
analysis_code = analysis_code.replace('skills_radar_chart_base64 = plot_skills_radar_chart(df_filtered, username)', 
                                      '# plot_skills_radar_chart will be called after behavioral_traits')
analysis_code = analysis_code.replace('behavioral_traits = analyze_behavioral_traits(df_filtered, username)',
                                      '''behavioral_traits = analyze_behavioral_traits(df_filtered, username)
    skills_radar_chart_base64 = plot_skills_radar_chart(behavioral_traits['skills'], username)''')

with open('whatsapp_analyzer/analysis_utils.py', 'w') as f:
    f.write(analysis_code)

