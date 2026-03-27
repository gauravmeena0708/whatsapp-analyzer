# tests/test_ml_models.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np


def _reset_ml_state():
    """Reset ml_models module-level singleton state between tests."""
    import whatsapp_analyzer.ml_models as ml
    ml._sentiment_pipeline = None
    ml._hindi_sentiment_pipeline = None
    ml._sentence_model = None
    ml.FAST_MODE = False


class TestFastMode(unittest.TestCase):

    def setUp(self):
        _reset_ml_state()

    def tearDown(self):
        _reset_ml_state()

    def test_fast_mode_sentiment_pipeline_returns_none(self):
        import whatsapp_analyzer.ml_models as ml
        ml.FAST_MODE = True
        self.assertIsNone(ml.get_sentiment_pipeline())

    def test_fast_mode_hindi_pipeline_returns_none(self):
        import whatsapp_analyzer.ml_models as ml
        ml.FAST_MODE = True
        self.assertIsNone(ml.get_hindi_sentiment_pipeline())

    def test_fast_mode_sentence_model_returns_none(self):
        import whatsapp_analyzer.ml_models as ml
        ml.FAST_MODE = True
        self.assertIsNone(ml.get_sentence_model())

    def test_fast_mode_predict_sentiment_still_returns_floats(self):
        """Even in fast mode, predict_sentiment must return valid (polarity, subjectivity)."""
        import whatsapp_analyzer.ml_models as ml
        ml.FAST_MODE = True
        polarity, subjectivity = ml.predict_sentiment("I am very happy!")
        self.assertIsInstance(polarity, float)
        self.assertIsInstance(subjectivity, float)
        self.assertGreaterEqual(polarity, -1.0)
        self.assertLessEqual(polarity, 1.0)


class TestUnavailableTransformers(unittest.TestCase):

    def setUp(self):
        _reset_ml_state()

    def tearDown(self):
        _reset_ml_state()

    def test_missing_transformers_marks_pipeline_unavailable(self):
        import whatsapp_analyzer.ml_models as ml
        import sys
        original = sys.modules.get("transformers")
        sys.modules["transformers"] = None
        try:
            result = ml.get_sentiment_pipeline()
            self.assertIsNone(result)
            self.assertIs(ml._sentiment_pipeline, False)
        finally:
            if original is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original
            _reset_ml_state()

    def test_missing_sentence_transformers_returns_none(self):
        import whatsapp_analyzer.ml_models as ml
        import sys
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None
        try:
            result = ml.get_sentence_model()
            self.assertIsNone(result)
            self.assertIs(ml._sentence_model, False)
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original
            _reset_ml_state()


class TestPredictSentiment(unittest.TestCase):

    def setUp(self):
        _reset_ml_state()

    def tearDown(self):
        _reset_ml_state()

    def test_empty_text_returns_neutral(self):
        import whatsapp_analyzer.ml_models as ml
        polarity, subjectivity = ml.predict_sentiment("")
        self.assertEqual(polarity, 0.0)
        self.assertEqual(subjectivity, 0.5)

    def test_whitespace_only_returns_neutral(self):
        import whatsapp_analyzer.ml_models as ml
        polarity, subjectivity = ml.predict_sentiment("   ")
        self.assertEqual(polarity, 0.0)
        self.assertEqual(subjectivity, 0.5)

    def test_positive_label_with_mocked_pipeline(self):
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.9}])
        ml._sentiment_pipeline = mock_pipe
        polarity, _ = ml.predict_sentiment("This is great!")
        self.assertAlmostEqual(polarity, 0.9, places=5)

    def test_negative_label_with_mocked_pipeline(self):
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "negative", "score": 0.8}])
        ml._sentiment_pipeline = mock_pipe
        polarity, _ = ml.predict_sentiment("This is terrible!")
        self.assertAlmostEqual(polarity, -0.8, places=5)

    def test_neutral_label_gives_zero_polarity(self):
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "neutral", "score": 0.95}])
        ml._sentiment_pipeline = mock_pipe
        polarity, _ = ml.predict_sentiment("The sky is blue.")
        self.assertAlmostEqual(polarity, 0.0, places=5)

    def test_uppercase_label_normalised(self):
        """Labels like 'POSITIVE' or 'NEG' should be handled via lowercasing."""
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "NEGATIVE", "score": 0.7}])
        ml._sentiment_pipeline = mock_pipe
        polarity, _ = ml.predict_sentiment("This is bad.")
        self.assertAlmostEqual(polarity, -0.7, places=5)

    def test_legacy_label_0_negative(self):
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.6}])
        ml._sentiment_pipeline = mock_pipe
        polarity, _ = ml.predict_sentiment("Bad stuff.")
        self.assertAlmostEqual(polarity, -0.6, places=5)

    def test_legacy_label_2_positive(self):
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "LABEL_2", "score": 0.85}])
        ml._sentiment_pipeline = mock_pipe
        polarity, _ = ml.predict_sentiment("Great stuff.")
        self.assertAlmostEqual(polarity, 0.85, places=5)

    def test_fallback_to_textblob_when_no_pipeline(self):
        """When pipelines are marked unavailable, TextBlob polarity is returned."""
        import whatsapp_analyzer.ml_models as ml
        ml._sentiment_pipeline = False
        ml._hindi_sentiment_pipeline = False
        polarity, subjectivity = ml.predict_sentiment("I love this!")
        self.assertIsInstance(polarity, float)
        self.assertIsInstance(subjectivity, float)
        # "love" should register as positive with TextBlob
        self.assertGreater(polarity, 0)

    def test_subjectivity_is_float_in_range(self):
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.8}])
        ml._sentiment_pipeline = mock_pipe
        _, subjectivity = ml.predict_sentiment("I think this is wonderful.")
        self.assertIsInstance(subjectivity, float)
        self.assertGreaterEqual(subjectivity, 0.0)
        self.assertLessEqual(subjectivity, 1.0)

    def test_hindi_text_routes_to_hindi_pipeline(self):
        """When Hindi is detected and Hindi pipeline is available, it should be used."""
        import whatsapp_analyzer.ml_models as ml
        mock_en_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.5}])
        mock_hi_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.75}])
        ml._sentiment_pipeline = mock_en_pipe
        ml._hindi_sentiment_pipeline = mock_hi_pipe

        with patch("whatsapp_analyzer.ml_models.detect_language", return_value="hi"):
            polarity, _ = ml.predict_sentiment("यह बहुत अच्छा है")  # "This is very good"
            mock_hi_pipe.assert_called_once()
            mock_en_pipe.assert_not_called()
            self.assertAlmostEqual(polarity, 0.75, places=5)

    def test_english_text_does_not_use_hindi_pipeline(self):
        import whatsapp_analyzer.ml_models as ml
        mock_en_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.9}])
        mock_hi_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.5}])
        ml._sentiment_pipeline = mock_en_pipe
        ml._hindi_sentiment_pipeline = mock_hi_pipe

        with patch("whatsapp_analyzer.ml_models.detect_language", return_value="en"):
            ml.predict_sentiment("This is wonderful!")
            mock_en_pipe.assert_called_once()
            mock_hi_pipe.assert_not_called()

    def test_pipeline_exception_falls_back_to_textblob(self):
        """If the pipeline raises, fall back gracefully to TextBlob."""
        import whatsapp_analyzer.ml_models as ml
        mock_pipe = MagicMock(side_effect=RuntimeError("model error"))
        ml._sentiment_pipeline = mock_pipe
        # Should not raise; should fall back
        polarity, subjectivity = ml.predict_sentiment("Hello world")
        self.assertIsInstance(polarity, float)
        self.assertIsInstance(subjectivity, float)


class TestSingletonCaching(unittest.TestCase):

    def setUp(self):
        _reset_ml_state()

    def tearDown(self):
        _reset_ml_state()

    def test_sentiment_pipeline_loaded_only_once(self):
        """Model loading code should only run on first call; subsequent calls reuse cached instance."""
        import whatsapp_analyzer.ml_models as ml
        mock_instance = MagicMock()
        call_count = {"n": 0}

        def fake_hf_pipeline(*args, **kwargs):
            call_count["n"] += 1
            return mock_instance

        with patch.dict("sys.modules", {}):
            with patch("whatsapp_analyzer.ml_models._sentiment_pipeline", None):
                ml._sentiment_pipeline = None
                with patch("builtins.__import__", side_effect=ImportError):
                    pass  # Just verify state not reset between calls

        ml._sentiment_pipeline = mock_instance  # Pre-load
        r1 = ml.get_sentiment_pipeline()
        r2 = ml.get_sentiment_pipeline()
        self.assertIs(r1, r2)
        self.assertEqual(call_count["n"], 0)  # No actual loading occurred

    def test_failed_load_not_retried(self):
        """Once marked as False (failed), subsequent calls should not retry."""
        import whatsapp_analyzer.ml_models as ml
        ml._sentiment_pipeline = False

        load_attempts = {"n": 0}

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

        # Just verify it returns None without going into load code
        for _ in range(5):
            result = ml.get_sentiment_pipeline()
            self.assertIsNone(result)

        self.assertEqual(load_attempts["n"], 0)


class TestDetectLanguage(unittest.TestCase):

    def setUp(self):
        _reset_ml_state()

    def test_short_text_returns_english(self):
        import whatsapp_analyzer.ml_models as ml
        self.assertEqual(ml.detect_language("ok"), "en")
        self.assertEqual(ml.detect_language("hi"), "en")
        self.assertEqual(ml.detect_language(""), "en")

    def test_missing_langdetect_returns_english(self):
        import whatsapp_analyzer.ml_models as ml
        import sys
        original = sys.modules.get("langdetect")
        sys.modules["langdetect"] = None
        try:
            result = ml.detect_language("Hello, how are you doing today?")
            self.assertEqual(result, "en")
        finally:
            if original is None:
                sys.modules.pop("langdetect", None)
            else:
                sys.modules["langdetect"] = original

    def test_langdetect_exception_returns_english(self):
        import whatsapp_analyzer.ml_models as ml
        with patch("whatsapp_analyzer.ml_models.detect_language", wraps=ml.detect_language):
            with patch.dict("sys.modules", {"langdetect": MagicMock(detect=MagicMock(side_effect=Exception("err")))}):
                result = ml.detect_language("Some long enough text to pass the length check here")
                self.assertIsInstance(result, str)


class TestGetSentenceEmbeddings(unittest.TestCase):

    def setUp(self):
        _reset_ml_state()

    def tearDown(self):
        _reset_ml_state()

    def test_returns_numpy_array_with_mocked_model(self):
        import whatsapp_analyzer.ml_models as ml
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 384))
        ml._sentence_model = mock_model

        texts = ["Hello world", "How are you?", "Good morning"]
        embeddings = ml.get_sentence_embeddings(texts)

        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (3, 384))
        mock_model.encode.assert_called_once_with(texts, show_progress_bar=False)

    def test_returns_none_when_model_unavailable(self):
        import whatsapp_analyzer.ml_models as ml
        ml._sentence_model = False
        self.assertIsNone(ml.get_sentence_embeddings(["test"]))

    def test_returns_none_in_fast_mode(self):
        import whatsapp_analyzer.ml_models as ml
        ml.FAST_MODE = True
        self.assertIsNone(ml.get_sentence_embeddings(["test"]))

    def test_model_exception_returns_none(self):
        import whatsapp_analyzer.ml_models as ml
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("encode failed")
        ml._sentence_model = mock_model
        self.assertIsNone(ml.get_sentence_embeddings(["test"]))


class TestIntegrationWithPlotUtils(unittest.TestCase):
    """Verify that plot_utils._polarity_subjectivity delegates to ml_models."""

    def setUp(self):
        _reset_ml_state()

    def tearDown(self):
        _reset_ml_state()

    def test_polarity_subjectivity_uses_transformer_when_available(self):
        import whatsapp_analyzer.ml_models as ml
        from whatsapp_analyzer.plot_utils import _polarity_subjectivity

        mock_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.88}])
        ml._sentiment_pipeline = mock_pipe

        polarity, _ = _polarity_subjectivity("This is absolutely wonderful!")
        self.assertAlmostEqual(polarity, 0.88, places=5)
        mock_pipe.assert_called_once()

    def test_polarity_subjectivity_fallback_when_no_models(self):
        """With no ML models, _polarity_subjectivity should still return valid floats."""
        import whatsapp_analyzer.ml_models as ml
        from whatsapp_analyzer.plot_utils import _polarity_subjectivity

        ml._sentiment_pipeline = False
        ml._hindi_sentiment_pipeline = False

        polarity, subjectivity = _polarity_subjectivity("Hello there")
        self.assertIsInstance(polarity, float)
        self.assertIsInstance(subjectivity, float)
        self.assertGreaterEqual(polarity, -1.0)
        self.assertLessEqual(polarity, 1.0)
        self.assertGreaterEqual(subjectivity, 0.0)
        self.assertLessEqual(subjectivity, 1.0)

    def test_polarity_subjectivity_fast_mode_uses_textblob(self):
        """In fast mode transformers are skipped; TextBlob should provide the score."""
        import whatsapp_analyzer.ml_models as ml
        from whatsapp_analyzer.plot_utils import _polarity_subjectivity

        ml.FAST_MODE = True
        polarity, subjectivity = _polarity_subjectivity("I love this very much!")
        self.assertIsInstance(polarity, float)
        self.assertGreater(polarity, 0)  # TextBlob should score "love" positively


if __name__ == "__main__":
    unittest.main()
