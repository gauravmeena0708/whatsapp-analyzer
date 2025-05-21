# This is an example script to demonstrate how to use the WhatsAppAnalyzer.

# Before running, make sure you have:
# 1. Exported your WhatsApp chat as a .txt file (without media).
#    See README.md for instructions on how to export your chat.
# 2. Updated the `chat_file_path` and `output_directory_path` variables below
#    to point to your chat file and desired output location, respectively.

# Import the WhatsAppAnalyzer class
from whatsapp_analyzer.analyzer import WhatsAppAnalyzer

# --- User Configuration ---
# TODO: Replace "YOUR_CHAT_FILE.txt" with the actual path to your WhatsApp chat export file.
# For example: "/home/user/downloads/whatsapp_chat.txt" or "C:\\Users\\YourName\\Documents\\whatsapp_chat.txt"
chat_file_path = "YOUR_CHAT_FILE.txt"

# TODO: Replace "YOUR_OUTPUT_DIRECTORY" with the actual path to the directory
# where you want the analysis report and any generated charts to be saved.
# For example: "/home/user/reports/whatsapp_analysis" or "C:\\Users\\YourName\\Documents\\MyReports"
# The directory will be created if it doesn't exist.
output_directory_path = "YOUR_OUTPUT_DIRECTORY"
# --- End User Configuration ---

# Step 1: Initialize the WhatsAppAnalyzer
# This creates an instance of the analyzer, configured with the path to your chat file
# and the directory where the output report will be saved.
print(f"Initializing analyzer with chat file: {chat_file_path} and output directory: {output_directory_path}...")
analyzer = WhatsAppAnalyzer(chat_file=chat_file_path, out_dir=output_directory_path)

# Step 2: Generate the analysis report
# This method processes the chat file and creates a report (e.g., a PDF file)
# in the specified output directory.
print("Generating report. This may take a few moments depending on the chat size...")
analyzer.generate_report()

# Confirmation message
print(f"Analysis complete. Report and related files should be generated in: {output_directory_path}")
# You can now open the output directory to view the generated report.
