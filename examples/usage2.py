# usage2.py (in the examples directory)
from whatsapp_analyzer.analyzer import WhatsAppAnalyzer

# Basic usage:
analyzer = WhatsAppAnalyzer(chat_file="../data/whatsapp_chat.txt", out_dir="../data")
analyzer.generate_report()  # Generates reports for all users with default settings

# analyzer.generate_report(users=["User 1", "User 2"]) # If you want to test specific users