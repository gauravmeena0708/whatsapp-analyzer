import re

with open('whatsapp_analyzer/analysis_utils.py', 'r') as f:
    content = f.read()

new_insights_logic = '''
    # VAD & Sarcasm Hints
    if traits.get('avg_arousal', 0.5) > 0.6:
        insights.append("Communication style is high-energy and active.")
    if traits.get('avg_dominance', 0.5) > 0.6:
        insights.append("Uses assertive and dominant language patterns.")
    if traits.get('sarcasm_count', 0) > 0:
        insights.append(f"Detected {traits['sarcasm_count']} instances of likely sarcasm (negative sentiment with playful emojis).")
'''

content = content.replace('return "<br/>".join(insights)', new_insights_logic + '\n    return "<br/>".join(insights)')

with open('whatsapp_analyzer/analysis_utils.py', 'w') as f:
    f.write(content)
