import re

with open('whatsapp_analyzer/ml_models.py', 'r') as f:
    content = f.read()

vad_sarcasm_logic = """
def analyze_vad(text):
    \"\"\"
    Simple lexical VAD (Valence, Arousal, Dominance) heuristic.
    In a real scenario, this would use a lexicon like NRC VAD.
    Returns (valence, arousal, dominance) in [0, 1].
    \"\"\"
    text = str(text).lower()
    # Very simple heuristics for demo purposes
    valence = 0.5
    arousal = 0.5
    dominance = 0.5
    
    # High arousal markers: exclamations, caps, certain words
    if '!' in text: arousal += 0.1
    if text.isupper() and len(text) > 5: arousal += 0.2
    
    # Arousal words
    active_words = {'wow', 'amazing', 'urgent', 'quickly', 'immediately', 'party', 'fire'}
    if any(w in text for w in active_words): arousal += 0.1
    
    # Valence is already handled by predict_sentiment, but we can refine
    # Dominance: use of assertive language
    assertive_words = {'must', 'need', 'will', 'do', 'stop', 'listen', 'command'}
    if any(w in text for w in assertive_words): dominance += 0.1
    
    return min(max(valence, 0), 1), min(max(arousal, 0), 1), min(max(dominance, 0), 1)

def detect_sarcasm(text, sentiment_score):
    \"\"\"
    Heuristic sarcasm detection: Negative sentiment + playful emojis.
    Returns boolean and confidence.
    \"\"\"
    text = str(text).lower()
    playful_emojis = {'😜', '😂', '🤣', '🤡', '😏', '🙃'}
    
    has_playful_emoji = any(e in text for e in playful_emojis)
    
    # Sarcasm candidate: negative polarity but has playful emojis
    if sentiment_score < -0.2 and has_playful_emoji:
        return True, 0.7
    
    return False, 0.0
"""

if 'def analyze_vad' not in content:
    content += vad_sarcasm_logic

with open('whatsapp_analyzer/ml_models.py', 'w') as f:
    f.write(content)
