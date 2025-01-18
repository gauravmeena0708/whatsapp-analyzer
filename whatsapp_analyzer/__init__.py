# whatsapp_analyzer/__init__.py
import nltk

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
