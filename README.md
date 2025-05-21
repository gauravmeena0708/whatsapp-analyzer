# Project Overview

This project is an enhanced analyzer for WhatsApp group chats. It provides insights into chat activity, such as message counts per user, most active users, common words used, and trends over time. The analyzer processes exported WhatsApp chat files and generates a comprehensive report.

## Installation

    pip install whatsapp-groupchat-analyzer

Alternatively, for development or to manage dependencies in a virtual environment, you can use the provided `requirements.txt` or `environment.yml` files.

For proper rendering of emojis in the report, it's recommended to have an emoji-supporting font installed on your system (e.g., 'Noto Color Emoji' on Linux, or standard system fonts on Windows/macOS). Missing emoji fonts might result in emojis being displayed as placeholders.

## How to Export WhatsApp Chat

To get your chat file:
1. Open the WhatsApp chat (group or individual) you want to analyze.
2. Tap on the three dots (menu) > More > Export chat.
3. Choose 'Without Media' to get a .txt file.
4. Save or send this .txt file to the environment where you are running this analyzer.

## How to run

    from whatsapp_analyzer.analyzer import WhatsAppAnalyzer

    # Replace 'path/to/your/whatsapp_chat.txt' with the actual path to your chat file
    # Replace 'path/to/your/output_directory' with where you want the report saved
    analyzer = WhatsAppAnalyzer(chat_file="path/to/your/whatsapp_chat.txt", out_dir="path/to/your/output_directory")
    analyzer.generate_report()
    print(f"Report generated in path/to/your/output_directory")

Ensure the `chat_file` is a plain text (.txt) file exported from WhatsApp (typically from the 'Export chat' feature on mobile, without media).
The `generate_report()` method will create a PDF report (and potentially other files like charts) in the specified `out_dir`.

## Contributing

Contributions are welcome! If you'd like to improve the analyzer, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Add tests for your changes if applicable.
5. Submit a pull request.

## License

This project is licensed under the terms of the LICENSE file.
