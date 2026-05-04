# whatsapp_analyzer/ml_models.py
"""
Optional ML model integrations for Phase 0.
All models are lazy-loaded singletons. Set FAST_MODE = True to skip model inference.

Dependencies (all optional):
    transformers          - for Twitter-RoBERTa and XLM-RoBERTa sentiment
    sentence-transformers - for MiniLM sentence embeddings
    langdetect            - for language detection
"""
import os

# Set to True via run.py --fast flag to skip all model inference
FAST_MODE = False
LOCAL_SUMMARY_MODEL_NAME = os.getenv("WHATSAPP_ANALYZER_LOCAL_SUMMARY_MODEL", "").strip() or None

# Lazy singleton state: None = not yet attempted, False = unavailable/failed
_sentiment_pipeline = None       # cardiffnlp/twitter-roberta-base-sentiment-latest
_hindi_sentiment_pipeline = None  # cardiffnlp/twitter-xlm-roberta-base-sentiment
_sentence_model = None            # all-MiniLM-L6-v2
_summary_pipeline = None
_summary_pipeline_model_name = None

# Map model output labels to polarity in [-1.0, 1.0]
_LABEL_TO_SIGN = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "pos": 1.0,
    "neg": -1.0,
    "neu": 0.0,
    "label_2": 1.0,   # original twitter-roberta: LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
    "label_1": 0.0,
    "label_0": -1.0,
}


def get_sentiment_pipeline():
    """
    Lazy-load Twitter-RoBERTa English sentiment pipeline.
    Returns None in fast mode or if transformers is unavailable.
    """
    global _sentiment_pipeline
    if FAST_MODE or _sentiment_pipeline is False:
        return None
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True,
                max_length=512,
            )
        except Exception:
            _sentiment_pipeline = False
            return None
    return _sentiment_pipeline


def get_hindi_sentiment_pipeline():
    """
    Lazy-load multilingual XLM-RoBERTa for Hindi/Hinglish sentiment.
    Returns None in fast mode or if transformers is unavailable.
    """
    global _hindi_sentiment_pipeline
    if FAST_MODE or _hindi_sentiment_pipeline is False:
        return None
    if _hindi_sentiment_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            _hindi_sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                truncation=True,
                max_length=512,
            )
        except Exception:
            _hindi_sentiment_pipeline = False
            return None
    return _hindi_sentiment_pipeline


def get_sentence_model():
    """
    Lazy-load MiniLM sentence transformer (~80MB, CPU-friendly).
    Returns None in fast mode or if sentence-transformers is unavailable.
    """
    global _sentence_model
    if FAST_MODE or _sentence_model is False:
        return None
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _sentence_model = False
            return None
    return _sentence_model


def set_local_summary_model(model_name):
    global LOCAL_SUMMARY_MODEL_NAME, _summary_pipeline, _summary_pipeline_model_name
    LOCAL_SUMMARY_MODEL_NAME = str(model_name).strip() or None
    _summary_pipeline = None
    _summary_pipeline_model_name = None


def get_local_summary_pipeline():
    """
    Lazy-load an optional local instruct model for summary generation.
    Returns None unless a model name was explicitly configured.
    """
    global _summary_pipeline, _summary_pipeline_model_name

    if FAST_MODE or not LOCAL_SUMMARY_MODEL_NAME:
        return None
    if _summary_pipeline is False and _summary_pipeline_model_name == LOCAL_SUMMARY_MODEL_NAME:
        return None
    if _summary_pipeline is not None and _summary_pipeline_model_name == LOCAL_SUMMARY_MODEL_NAME:
        return _summary_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        _summary_pipeline = hf_pipeline(
            "text-generation",
            model=LOCAL_SUMMARY_MODEL_NAME,
            tokenizer=LOCAL_SUMMARY_MODEL_NAME,
            truncation=True,
        )
        _summary_pipeline_model_name = LOCAL_SUMMARY_MODEL_NAME
    except Exception:
        _summary_pipeline = False
        _summary_pipeline_model_name = LOCAL_SUMMARY_MODEL_NAME
        return None
    return _summary_pipeline


def detect_language(text):
    """
    Detect language of text. Returns ISO 639-1 code (e.g. 'en', 'hi').
    Defaults to 'en' on short text or if langdetect is unavailable.
    """
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def predict_sentiment(text):
    """
    Predict sentiment using the best available model.

    Returns (polarity, subjectivity) compatible with TextBlob's API:
        polarity    : float in [-1.0, 1.0]
        subjectivity: float in [0.0, 1.0]  (TextBlob if available, else 0.5)

    Model selection priority:
        1. XLM-RoBERTa (if Hindi detected and Hindi pipeline available)
        2. Twitter-RoBERTa (English, if available)
        3. TextBlob (fallback)
        4. Simple word-list heuristic (last resort)
    """
    text = str(text).strip()
    if not text:
        return 0.0, 0.5

    # Subjectivity: TextBlob still best for this dimension
    subjectivity = 0.5
    try:
        from textblob import TextBlob
        subjectivity = float(TextBlob(text).sentiment.subjectivity)
    except Exception:
        pass

    # Choose pipeline: route Hindi text to multilingual model when available
    pipeline = get_sentiment_pipeline()
    hindi_pipeline = get_hindi_sentiment_pipeline()
    if hindi_pipeline is not None:
        lang = detect_language(text)
        if lang == "hi":
            pipeline = hindi_pipeline

    if pipeline is not None:
        try:
            result = pipeline(text)[0]
            label = result["label"].lower()
            score = float(result["score"])
            polarity = _LABEL_TO_SIGN.get(label, 0.0) * score
            return polarity, subjectivity
        except Exception:
            pass

    # Fallback to TextBlob polarity
    try:
        from textblob import TextBlob
        polarity = float(TextBlob(text).sentiment.polarity)
        return polarity, subjectivity
    except Exception:
        pass

    # Last resort: simple word-list heuristic
    positive_words = {"good", "great", "happy", "love", "excellent", "awesome", "nice", "best"}
    negative_words = {"bad", "sad", "hate", "angry", "terrible", "awful", "worst"}
    words = text.lower().split()
    if not words:
        return 0.0, subjectivity
    pos = sum(w in positive_words for w in words)
    neg = sum(w in negative_words for w in words)
    polarity = (pos - neg) / max(len(words), 1)
    return polarity, subjectivity


def get_sentence_embeddings(texts):
    """
    Compute sentence embeddings using MiniLM.
    Returns numpy array of shape (len(texts), 384), or None if model unavailable.
    """
    model = get_sentence_model()
    if model is None:
        return None
    try:
        return model.encode(texts, show_progress_bar=False)
    except Exception:
        return None


def generate_local_summary(prompt, max_new_tokens=120):
    """
    Generate a concise summary with an optional local instruct model.
    Returns None if no local summary model is configured or inference fails.
    """
    pipeline = get_local_summary_pipeline()
    if pipeline is None:
        return None

    try:
        result = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            return_full_text=False,
            pad_token_id=getattr(getattr(pipeline, "tokenizer", None), "eos_token_id", None),
        )
        if not result:
            return None
        text = result[0].get("generated_text", "").strip()
        return text or None
    except Exception:
        return None

def analyze_vad(text):
    """
    Simple lexical VAD (Valence, Arousal, Dominance) heuristic.
    In a real scenario, this would use a lexicon like NRC VAD.
    Returns (valence, arousal, dominance) in [0, 1].
    """
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
    """
    Heuristic sarcasm detection: Negative sentiment + playful emojis.
    Returns boolean and confidence.
    """
    text = str(text).lower()
    playful_emojis = {'😜', '😂', '🤣', '🤡', '😏', '🙃'}
    
    has_playful_emoji = any(e in text for e in playful_emojis)
    
    # Sarcasm candidate: negative polarity but has playful emojis
    if sentiment_score < -0.2 and has_playful_emoji:
        return True, 0.7
    
    return False, 0.0
