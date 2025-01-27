import nltk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# Font Initialization with System-Wide Check
def initialize_fonts():
    """
    Initializes and sets the fonts for matplotlib, checking for system-installed fonts.
    """
    font_issue = False  # Flag to track if there are any issues with fonts

    # Emoji-compatible fonts to check
    emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji", "Noto Emoji"]

    # System-wide font list
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    # Find available emoji-compatible font
    selected_emoji_font = None
    for font in emoji_fonts:
        if font in available_fonts:
            selected_emoji_font = font
            break

    # Set font families
    if selected_emoji_font:
        plt.rcParams["font.family"] = [selected_emoji_font, "sans-serif"]
    else:
        warnings.warn(
            "No emoji-compatible font found. Using default 'sans-serif'. "
            "Install 'Segoe UI Emoji', 'Apple Color Emoji', or 'Noto Emoji' for better emoji support."
        )
        plt.rcParams["font.family"] = ["sans-serif"]

    # Check for system-wide Roboto font
    if "Roboto" in available_fonts:
        plt.rcParams["font.family"].insert(0, "Roboto")

initialize_fonts()

def ensure_nltk_resources(resources):
    """
    Ensure that the required NLTK resources are available.
    Downloads the resources only if they are not already present.

    Args:
        resources (list of tuples): List of resources to check, each as a tuple
            (resource_name, resource_type).
    """
    for resource_name, _ in resources:
            nltk.download(resource_name)

# List of required NLTK resources
required_resources = [
    ('punkt', 'tokenizers'),
    ('stopwords', 'corpora'),
    ('vader_lexicon', 'sentiment'),
    ('averaged_perceptron_tagger', 'taggers'),
    ('wordnet', 'corpora'),
]

# Ensure resources are available
ensure_nltk_resources(required_resources)
