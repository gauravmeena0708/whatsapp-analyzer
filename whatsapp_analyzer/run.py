import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import sys
from .analyzer import WhatsAppAnalyzer

def setup_fonts():
    """Setup emoji-compatible fonts for matplotlib."""
    # Enumerate fonts per-file so a single malformed TTF doesn't abort the whole scan.
    available_fonts = set()
    for fp in fm.findSystemFonts(fontext='ttf'):
        try:
            available_fonts.add(fm.FontProperties(fname=fp).get_name())
        except Exception:
            continue

    emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji"]
    selected_font = next((f for f in emoji_fonts if f in available_fonts), None)

    if selected_font:
        plt.rcParams["font.family"] = [selected_font, "DejaVu Sans", "sans-serif"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

def main():
    parser = argparse.ArgumentParser(description="WhatsApp Chat Analyzer")
    parser.add_argument("chat_file", help="Path to the WhatsApp chat export file (txt)")
    parser.add_argument("-o", "--output", default="reports", help="Directory to save generated reports (default: reports)")
    parser.add_argument("-u", "--users", nargs="+", help="Specific users to generate reports for (default: all)")
    parser.add_argument("--fast", action="store_true", help="Skip ML model inference (faster processing, falls back to TextBlob for sentiment)")
    parser.add_argument(
        "--local-summary-model",
        help="Optional local instruct model for monthly summaries, e.g. Qwen/Qwen2.5-7B-Instruct",
    )

    args = parser.parse_args()

    if not os.path.exists(args.chat_file):
        print(f"Error: Chat file '{args.chat_file}' not found.")
        sys.exit(1)

    try:
        from whatsapp_analyzer import ml_models
        if args.fast:
            ml_models.FAST_MODE = True
        if args.local_summary_model:
            ml_models.set_local_summary_model(args.local_summary_model)
    except ImportError:
        pass

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
