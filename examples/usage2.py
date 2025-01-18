
from whatsapp_analyzer.analyzer import WhatsAppAnalyzer
analyzer = WhatsAppAnalyzer(chat_file="../data/whatsapp_chat.txt", out_dir="../data")
analyzer.generate_report() 