# Introduction

This is second repo for whatsapp group analyzer. The complete usage example is given in examples/usage1.py

## How to run

    from whatsapp_analyzer.analyzer import WhatsAppAnalyzer
    analyzer = WhatsAppAnalyzer(chat_file="../data/whatsapp_chat.txt", out_dir="../data")
    analyzer.generate_report()
