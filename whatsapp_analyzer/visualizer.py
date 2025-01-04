# whatsapp_analyzer/visualizer.py

# You might want to make visualization a separate package
# that depends on your core whatsapp_analyzer package.

import matplotlib.pyplot as plt
import seaborn as sns

class ChatVisualizer:
    """
    Visualizes WhatsApp chat analysis results.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def plot_message_counts(self):
        """
        Plots a bar chart of message counts per sender.
        """
        counts = self.analyzer.message_counts()
        # ... use matplotlib or seaborn to create the bar chart ...

    def plot_activity_over_time(self):
        """
        Plots a line chart of message activity over time.
        """
        activity = self.analyzer.activity_by_date()  # or activity_by_time()
        # ... use matplotlib or seaborn to create the line chart ...

    # ... other visualization methods ...