import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import sys
from .analyzer import WhatsAppAnalyzer

def setup_fonts():
    """Setup emoji-compatible fonts for matplotlib."""
    try:
        available_fonts = {fm.FontProperties(fname=fp).get_name() for fp in fm.findSystemFonts(fontext='ttf')}
        emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji"]
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
    except Exception as e:
        warnings.warn(f"Font setup failed: {e}. Falling back to default fonts.")
        plt.rcParams["font.family"] = ["sans-serif"]

def main():
    parser = argparse.ArgumentParser(description="WhatsApp Chat Analyzer")
    parser.add_argument("chat_file", help="Path to the WhatsApp chat export file (txt)")
    parser.add_argument("-o", "--output", default="reports", help="Directory to save generated reports (default: reports)")
    parser.add_argument("-u", "--users", nargs="+", help="Specific users to generate reports for (default: all)")

    args = parser.parse_args()

    if not os.path.exists(args.chat_file):
        print(f"Error: Chat file '{args.chat_file}' not found.")
        sys.exit(1)

    setup_fonts()

    print(f"Analyzing {args.chat_file}...")
    try:
        analyzer = WhatsAppAnalyzer(chat_file=args.chat_file, out_dir=args.output)
        analyzer.generate_report(users=args.users)
        print(f"\nAll reports have been generated and saved in the '{args.output}' directory.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
