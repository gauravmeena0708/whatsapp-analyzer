import os # For os.path.join and os.makedirs if out_dir needs creation by analyzer
import matplotlib.pyplot as plt # For plt.rcParams
import matplotlib.font_manager as fm # For font management
import warnings # For warnings.warn

from .analyzer import WhatsAppAnalyzer # Import the main class

# Get all installed font names
available_fonts = {fm.FontProperties(fname=fp).get_name() for fp in fm.findSystemFonts(fontext='ttf')} # Added fontext for robustness

# Add an emoji-compatible font if available
emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji"] # Added Noto
selected_font = None

for font in emoji_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams["font.family"] = [selected_font, "Roboto", "DejaVu Sans", "sans-serif"]
else:
    warnings.warn(
        "No emoji-compatible font found. Install 'Segoe UI Emoji', 'Apple Color Emoji', or 'Noto Color Emoji' for full emoji support."
    )
    plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans", "sans-serif"]

def main():
    """Main function to run the WhatsApp chat analysis."""

    chat_file = "../data/whatsapp_chat.txt"  # Replace with your chat file path
    output_dir = "../data"                  # Specify output directory for reports

    # Initialize WhatsAppAnalyzer
    analyzer = WhatsAppAnalyzer(chat_file=chat_file, out_dir=output_dir)

    # Generate reports for all users found in the chat file
    # The generate_report method handles iterating through users and saving files.
    analyzer.generate_report()

    print(f"All reports have been generated and saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()