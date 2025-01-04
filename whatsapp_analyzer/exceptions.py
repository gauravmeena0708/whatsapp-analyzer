# whatsapp_analyzer/exceptions.py

class ChatFileNotFoundError(FileNotFoundError):
    """Custom exception raised when the chat file is not found."""
    pass  # No custom behavior, just inherits FileNotFoundError

class ParseError(Exception):
    """Custom exception raised for errors during parsing."""
    pass  # No custom behavior, just inherits Exception