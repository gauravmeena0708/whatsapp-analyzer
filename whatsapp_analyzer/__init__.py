"""Public package interface for whatsapp_analyzer.

Keep package import lightweight so modules like ``parser`` can be imported
without pulling in optional plotting/NLP/report dependencies.
"""

__all__ = ["WhatsAppAnalyzer", "anonymize", "df_basic_cleanup"]


def __getattr__(name):
    if name == "WhatsAppAnalyzer":
        from .analyzer import WhatsAppAnalyzer

        return WhatsAppAnalyzer
    if name in {"anonymize", "df_basic_cleanup"}:
        from .utils import anonymize, df_basic_cleanup

        return {"anonymize": anonymize, "df_basic_cleanup": df_basic_cleanup}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
