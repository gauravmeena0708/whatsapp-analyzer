# Introduction

This is second repo for whatsapp group analyzer. The exaple usage is as following

## Installation

    pip install whatsapp-groupchat-analyzer

it is recommended to have emoji fonts installed in system

## How to run

    from whatsapp_analyzer.analyzer import WhatsAppAnalyzer
    analyzer = WhatsAppAnalyzer(chat_file="../data/whatsapp_chat.txt", out_dir="../data")
    analyzer.generate_report()
