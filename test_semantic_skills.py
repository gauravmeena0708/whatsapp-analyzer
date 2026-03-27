import sys
import os

# Add current dir to python path
sys.path.append(os.getcwd())

from whatsapp_analyzer.ml_models import get_sentence_embeddings
from whatsapp_analyzer.constants import skill_keywords
import numpy as np

# Compute skill embeddings
skill_embeddings = {}
for skill, keywords in skill_keywords.items():
    # We can embed the skill name and keywords
    text = f"{skill}: " + " ".join(keywords)
    emb = get_sentence_embeddings([text])
    if emb is not None:
        skill_embeddings[skill] = emb[0]

if not skill_embeddings:
    print("Models not available")
    sys.exit(0)

# Sample messages
messages = [
    "I can help coordinate the event and lead the team.",
    "The database server is down, let me debug the code.",
    "We need to work together and cooperate on this project.",
    "Please explain how the system works.",
    "Let's fix the bug in the algorithm."
]

msg_embs = get_sentence_embeddings(messages)

from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity(msg_embs, list(skill_embeddings.values()))
skill_names = list(skill_embeddings.keys())

for msg, score_row in zip(messages, scores):
    print(f"Message: {msg}")
    for skill, s in zip(skill_names, score_row):
        print(f"  {skill}: {s:.3f}")
    print()
