import re

# 1. Update plot_utils.py
with open('whatsapp_analyzer/plot_utils.py', 'r') as f:
    plot_code = f.read()

new_heatmap = '''def plot_interaction_matrix(matrix, username=None):
    if not matrix:
        return "<p class='text-center text-muted'>Interaction data unavailable.</p>"

    users = list(matrix.keys())
    if not users: return ""
    
    # If username is provided, we might want to highlight their row/col
    # but for simplicity we render the full matrix or a slice
    
    labels = users
    datasets = []
    
    # Chart.js Heatmap is tricky, we'll use a bubble chart or a colored grid if possible.
    # Actually, a Bar chart (stacked or group) showing "Who I interact with most" is better for UI.
    
    if username and username in matrix:
        # Show who 'username' interacts with
        interactions = matrix[username]
        sorted_inter = sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:10]
        if not sorted_inter: return "<p class='text-center text-muted'>No interactions found.</p>"
        
        target_users, counts = zip(*sorted_inter)
        
        config = {
            "type": "bar",
            "data": {
                "labels": list(target_users),
                "datasets": [{"label": "Interactions", "data": list(counts), "backgroundColor": "#25d366"}]
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"title": {"display": True, "text": f"Top Interactions for {username}"}}
            }
        }
        return render_chartjs(config)
    
    return "<p class='text-center text-muted'>Global matrix visualization pending.</p>"'''

if 'def plot_interaction_matrix' not in plot_code:
    plot_code += "\n\n" + new_heatmap

with open('whatsapp_analyzer/plot_utils.py', 'w') as f:
    f.write(plot_code)


# 2. Update analysis_utils.py
with open('whatsapp_analyzer/analysis_utils.py', 'r') as f:
    analysis_code = f.read()

social_logic = '''
def analyze_social_dynamics(df):
    """
    Analyze social dynamics like initiation rate and reciprocity.
    """
    users = [u for u in df['name'].unique() if u != 'System']
    if not users: return {'initiators': {}, 'interaction_matrix': {}}

    df_sorted = df.sort_values('date_time').copy()
    if df_sorted.empty: return {'initiators': {}, 'interaction_matrix': {}}
    
    # 1. Initiation Rate
    df_sorted['time_gap'] = df_sorted['date_time'].diff()
    df_sorted['is_initiation'] = (df_sorted['time_gap'] > pd.Timedelta(hours=6))
    # First message is initiation
    df_sorted.iloc[0, df_sorted.columns.get_loc('is_initiation')] = True
    
    initiators = df_sorted[df_sorted['is_initiation']]['name'].value_counts().to_dict()
    
    # 2. Reciprocity / Interaction Matrix
    interaction_matrix = {u: {v: 0 for v in users} for u in users}
    
    for i in range(1, len(df_sorted)):
        prev_row = df_sorted.iloc[i-1]
        curr_row = df_sorted.iloc[i]
        
        if (curr_row['date_time'] - prev_row['date_time']) < pd.Timedelta(hours=1):
            u, v = curr_row['name'], prev_row['name']
            if u != 'System' and v != 'System' and u != v:
                if u in interaction_matrix and v in interaction_matrix[u]:
                    interaction_matrix[u][v] += 1
                
    return {
        'initiators': initiators,
        'interaction_matrix': interaction_matrix
    }
'''

if 'def analyze_social_dynamics' not in analysis_code:
    analysis_code = analysis_code.replace('from .plot_utils import (', social_logic + '\nfrom .plot_utils import (')

# Add plot_interaction_matrix to imports
analysis_code = analysis_code.replace('    plot_personality_radar\n)', '    plot_personality_radar,\n    plot_interaction_matrix\n)')

with open('whatsapp_analyzer/analysis_utils.py', 'w') as f:
    f.write(analysis_code)
