# whatsapp_analyzer/__init__.py
import nltk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt # Already imported
import warnings

available_fonts = {fm.FontProperties(fname=fp).get_name() for fp in fm.findSystemFonts()}
emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji"]
selected_font = None

for font in emoji_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams["font.family"] = [selected_font, "Roboto", "DejaVu Sans", "sans-serif"]
else:
    warnings.warn(
        "No emoji-compatible font found. Install 'Segoe UI Emoji', or 'Apple Color Emoji' for full emoji support."
    )
    plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans", "sans-serif"]

def ensure_nltk_resources(resources):
    """
    Ensure that the required NLTK resources are available.
    Downloads the resources only if they are not already present.

    Args:
        resources (list of tuples): List of resources to check, each as a tuple
            (resource_name, resource_type).
    """
    for resource_name, resource_type in resources:
        try:
            # Check if the resource exists
            nltk.data.find(f'{resource_type}/{resource_name}')
            print("NLTK resource found:", resource_name)
        except LookupError:
            # Download if the resource is missing
            print(f"Downloading NLTK resource: {resource_name}")
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
