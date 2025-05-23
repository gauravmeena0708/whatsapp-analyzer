# WhatsApp Chat Analyzer

## Introduction

The WhatsApp Chat Analyzer is a Python project designed to parse and analyze WhatsApp chat exports. It generates detailed HTML reports for each user, providing statistics, various visualizations (like activity heatmaps, sentiment trends, word clouds), and behavioral insights based on the chat data. This tool helps users understand communication patterns, sentiment dynamics, language complexity, and more from their WhatsApp conversations.

## Features

*   **General Statistics:**
    *   Total messages, words, emojis, media files shared, and URLs.
    *   Counts of edited and deleted messages.
*   **Activity Analysis:**
    *   Activity heatmaps showing message frequency by day of the week and hour.
    *   Most active hours and overall activity trends.
    *   Average response time distribution.
*   **Sentiment Analysis:**
    *   Distribution of message sentiment (positive, negative, neutral).
    *   Sentiment trends over time.
    *   Sentiment bubble charts (polarity vs. subjectivity).
*   **Emotion Analysis:**
    *   Emotion trends (joy, surprise, sadness, anger, neutral) over time based on message content.
*   **Language Analysis:**
    *   Word clouds to visualize the most frequent words.
    *   Commonly used N-grams (unigrams, bigrams, trigrams).
    *   Language complexity metrics (average word length, average sentence length).
    *   Vocabulary diversity.
    *   Part-of-Speech (POS) tagging distribution.
*   **Behavioral Insights:**
    *   Keyword-based analysis for skills (e.g., communication, technical, leadership).
    *   Detection of Hindi abusive words.
    *   Generated textual insights based on analyzed traits.
*   **User Relationship Graphs:**
    *   Network graphs visualizing interactions between users.
*   **Individual HTML Reports:**
    *   Generates a separate, comprehensive HTML report for each user in the chat.

## Installation

1.  **Install the package:**
    ```bash
    pip install whatsapp-groupchat-analyzer
    ```
2.  **Python Version:**
    *   Recommended: Python 3.8 or higher.
3.  **Emoji Fonts:**
    *   For the best display of emojis in generated reports and visualizations, it is highly recommended to have emoji-supporting fonts installed on your system. Examples include "Segoe UI Emoji" (Windows), "Apple Color Emoji" (macOS), or "Noto Color Emoji" (Linux).

## How to Use

### As a Python library

This is the primary way to use the analyzer.

```python
from whatsapp_analyzer.analyzer import WhatsAppAnalyzer

# Initialize with the path to your WhatsApp chat file and desired output directory
# Replace with the actual path to your exported chat file (usually a .txt file)
chat_file_path = "path/to/your/whatsapp_chat.txt" 
output_directory_path = "path/to/output_directory" # Reports will be saved here

analyzer = WhatsAppAnalyzer(chat_file=chat_file_path, 
                            out_dir=output_directory_path)

# Generate reports for all users found in the chat
analyzer.generate_report()

# To generate reports for specific users (optional):
# analyzer.generate_report(users=["User1", "User2"]) 
# Replace "User1", "User2" with actual names as they appear in the chat.
```

### As an example script

The project includes an example script `whatsapp_analyzer/run.py` that demonstrates how to use the `WhatsAppAnalyzer` class.

1.  **Navigate to the project directory.**
2.  **Modify paths in `run.py` (if necessary):**
    *   Open `whatsapp_analyzer/run.py`.
    *   Change the `chat_file` and `output_dir` variables to point to your WhatsApp chat export file and your desired output folder, respectively.
3.  **Run the script:**
    If you are in the parent directory of `whatsapp_analyzer` (e.g., the root of the cloned repository), you can run it as a module:
    ```bash
    python -m whatsapp_analyzer.run
    ```

## Understanding the Output

The analyzer generates an HTML report for each user (e.g., `UserName_report.html`) in the specified output directory. Each report includes:

*   **Profile Summary:** Basic user information (placeholder for now).
*   **Key Statistics:** Tables with various counts and averages (total messages, words, emojis, average message length, etc.).
*   **Common Words:** Lists of most frequent unigrams, bigrams, and trigrams, plus detected Hindi abuse words.
*   **Visualizations:** A rich set of charts and graphs embedded directly in the report, including:
    *   Activity Heatmap
    *   Most Active Hours
    *   Response Time Distribution
    *   Sentiment Over Time & Distribution
    *   Emotion Trends
    *   Emoji Usage
    *   Word Cloud
    *   Language Complexity & Vocabulary Diversity plots
    *   User Relationship Graph (same for all users, shows overall chat interaction)
    *   Skills Radar Chart
*   **Behavioral Insights:** A textual summary of potential behavioral traits inferred from the analysis.

## Troubleshooting/Notes

*   **Emoji Display:** For the best visual experience, ensure you have emoji-supporting fonts installed on your system (e.g., Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji).
*   **Chat File Format:** The analyzer is designed to work with standard WhatsApp chat export files, which are typically `.txt` files. Encrypted chat backups are not supported.
*   **Large Chats:** Analysis of very large chat files can be memory and time-intensive.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/issue-number`.
3.  **Make your changes** and commit them with clear, descriptive messages.
4.  **Push your changes** to your forked repository.
5.  **Submit a Pull Request** to the main repository's `main` or `develop` branch.

Please also feel free to open an issue if you find a bug or have a suggestion for a new feature.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
(Note: You'll need to create a `LICENSE` file in your repository containing the MIT License text if you haven't already.)
