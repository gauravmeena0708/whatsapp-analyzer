# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analyzer
python -m whatsapp_analyzer.run chat.txt -o reports/
python -m whatsapp_analyzer.run chat.txt -o reports/ -u "User1" "User2"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_parser.py

# Run a single test
pytest tests/test_parser.py::test_single_line_message

# Build for PyPI
python -m build
```

## Architecture

The pipeline flows: **Parse → Clean → Feature Extract → Analyze → Plot → Generate HTML**

### Module Responsibilities

- **parser.py** — Regex-based parsing of WhatsApp `.txt` exports into a DataFrame with columns `t` (timestamp), `name`, `message`. Handles multiline messages via preprocessing.

- **utils.py** — `df_basic_cleanup()` orchestrates all feature extraction: emojis, URLs, datetime features (hour, day, week, month), word/emoji/URL counts, and message type flags (media omitted, edited, deleted). Called once on the parsed DataFrame before analysis.

- **analysis_utils.py** — `basic_stats()` computes per-user and group-level statistics. Also handles sentiment/emotion scoring, n-gram calculation, behavioral trait analysis (keyword matching against `constants.py`), and user relationship graphs.

- **plot_utils.py** — Generates all charts (heatmaps, word clouds, network graphs, sentiment trends, emoji charts, etc.) and returns them as base64-encoded strings for embedding directly into HTML. Uses LRU cache on `clean_message()`.

- **analyzer.py** — Orchestrator. Initializes parser, calls `df_basic_cleanup()`, then drives `generate_report()` which writes per-user HTML reports and a group `index.html`.

- **constants.py** — Contains stop words (English + Hinglish), skill keyword dictionaries (5 categories), a Hindi abusive words list, and the full Bootstrap-based HTML templates for individual and index reports.

- **pdf_generator.py** — Converts HTML reports to PDF via `pdfkit` (requires `wkhtmltopdf`). Uses BeautifulSoup to modify HTML before rendering.

### Key Design Choices

- All charts are embedded as base64 in HTML — no external image files needed in the output directory.
- The `__init__.py` uses lazy-loading (`__getattr__`) so importing the package is lightweight.
- Security: all user-generated content is passed through `html.escape()` before insertion into HTML templates; output filenames are sanitized with regex.
- Optional dependencies (`emoji`, `nltk`, `textblob`, `wordcloud`) are imported inside try-except blocks — missing them degrades features gracefully rather than crashing.
- Emoji extraction uses the `emoji` library's native method; falls back to `regex` (grapheme-aware) if needed.
