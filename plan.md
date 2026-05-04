# WhatsApp Chat Analyzer: Advanced Psychological Profiling Plan

This plan outlines the evolution of the WhatsApp Chat Analyzer from a statistical tool to a deep psychological and personality profiling engine.

---

## Phase 0: Open Source Model Integration (Foundation Upgrade)

Replace weak baseline models with lightweight, task-appropriate open source models before building higher-level features. All models should be **optional dependencies** — wrap in try-except so the tool works without them. Add a `--fast` flag to `run.py` to skip model inference for large chats.

### 0.1 Sentiment: Replace TextBlob
- [x] Implement `cardiffnlp/twitter-roberta-base-sentiment-latest` via `ml_models.get_sentiment_pipeline()`
- [x] Drop-in replacement in `plot_utils._polarity_subjectivity()` — routes through `ml_models.predict_sentiment()`
- [x] Fallback chain: transformer → TextBlob → word-list heuristic

### 0.2 Hindi/Hinglish Sentiment
- [x] Implement `cardiffnlp/twitter-xlm-roberta-base-sentiment` via `ml_models.get_hindi_sentiment_pipeline()`
- [x] Language detection per message via `langdetect` (`ml_models.detect_language()`)
- [x] Auto-route Hindi text (`lang == 'hi'`) to multilingual pipeline in `predict_sentiment()`

### 0.3 Semantic Similarity & Topic Clustering
- [x] Implement `all-MiniLM-L6-v2` via `ml_models.get_sentence_model()` and `get_sentence_embeddings()`
- [ ] Use embeddings for conversation thread detection in `analysis_utils.py`
- [ ] Replace keyword matching in skill/trait detection with semantic similarity

### 0.4 Infrastructure
- [x] All models lazy-loaded as singletons (`None` = not loaded, `False` = unavailable)
- [x] `FAST_MODE` flag in `ml_models.py` skips all model loading
- [x] `--fast` CLI flag in `run.py` sets `ml_models.FAST_MODE = True`
- [x] 31 tests in `tests/test_ml_models.py` covering fast mode, fallbacks, label mapping, Hindi routing, caching

### 0.5 What to Avoid
- LLaMA/Mistral — too large and slow for per-message inference at chat scale
- Full BERT for sarcasm — ~400MB, marginal gain, no good Hinglish sarcasm model exists yet
- Plotly server-side rendering — output is static HTML; use Chart.js instead (see Section 5)

---

## 1. Personality Mapping (The OCEAN Model)

Implement the **Big Five Personality Traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) using linguistic markers.

- [x] **Extraversion**: Score from message frequency, first-person plural pronouns ("we", "us"), emoji/exclamation density
- [x] **Agreeableness**: Score from positive emotion words, low abuse count, inclusive language
- [x] **Neuroticism**: Score from negative emotion words, first-person singular pronouns ("I", "me", "mine")
- [x] **Conscientiousness**: Score from average message length, formal/structured language style
- [x] **Openness**: Score from lexical diversity and abstract/novel concept mentions
- [x] Normalize all OCEAN scores as percentiles within the group (not raw counts)
- [x] Add minimum message threshold (suggested: 50) before showing scores

---

## 2. Social Dynamics & Relationship Mapping

Go beyond simple "mentions" to understand the hierarchy and bonds within the group.

- [x] **Reciprocity Score**: Matrix of reply balance between User A and User B
- [x] **Initiation Rate**: Identify who starts conversations vs. who only joins existing ones
- [x] **Response Latency Matrix**: Bond strength measured by how quickly User A responds to User B vs. others
- [x] **Influence Score**: How much a user's message triggers a chain reaction of replies

---

## 3. Automated Group Role Identification

Assign roles based on data-driven psychological signatures.

- [x] **The Mediator**: High response rate, high agreeableness, inclusive language
- [x] **The Energizer**: High emoji density, most active during peak hours, high exclamation count
- [x] **The Lurker**: High participation in active periods with low word count/frequency
- [x] **The Expert**: High technical keyword matches, long/structured sentences, low response latency
- [x] **The Night Owl**: High percentage of messages during late-night hours
- [x] **The Social Butterfly**: High Extraversion and a wide web of connections

---

## 4. Advanced Sentiment & Contextual Intelligence

- [x] **VAD Model**: Implement Valence-Arousal-Dominance to distinguish "Excited" (High Arousal) vs. "Calm" (Low Arousal) positivity — requires NRC VAD or ANEW lexicon
- [x] **Sarcasm & Banter Detection**: Correlate playful emojis (😜, 😂) with negative sentiment markers — mark as low-confidence/opt-in
- [x] **Conflict Heatmap**: Identify tension periods from sentiment drops and rapid-fire short messages

---

## 5. Technical & Interactive Enhancements

- [x] **Interactive Visualizations**: Replace static Matplotlib PNGs with **Chart.js** (preferred over Plotly — no server dependency, data embedded as JSON in static HTML)
- [x] **Comparative Benchmarking**: Add "User vs. Group Average" metrics to every report
- [x] **Temporal Evolution**: "Personality Over Time" graph — minimum 20 messages per month window required for stable scores

---

## 6. Implementation Strategy

- [x] **Update `analysis_utils.py`**: Add OCEAN scoring and relationship matrix logic
- [x] **Update `constants.py`**: Add Big Five weightings and psychological keyword lists
- [x] **Refactor `analyzer.py`**: Update reporting logic for group-wide comparative data
- [x] **Modernize `html_template`**: Integrate Chart.js for interactive dashboard
