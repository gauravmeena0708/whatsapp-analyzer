import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from whatsapp_analyzer.parser import Parser
from whatsapp_analyzer import utils
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import os
import emoji
import base64
from io import BytesIO
from functools import lru_cache
import networkx as nx
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Paths to font files
roboto_font_path = './reports/roboto.ttf'  # Path to Roboto
noto_emoji_font_path = './reports/noto.ttf'  # Path to Noto Color Emoji

# Add fonts explicitly
fm.fontManager.addfont(roboto_font_path)
fm.fontManager.addfont(noto_emoji_font_path)

# Load the fonts
font_prop_roboto = fm.FontProperties(fname=roboto_font_path)
#font_prop_emoji = fm.FontProperties(fname=noto_emoji_font_path)

# Set font families with fallback
plt.rcParams['font.family'] = [font_prop_roboto.get_name(), 'Noto Emoji', 'sans-serif' ]



# Download necessary NLTK data (only needed once)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Load and clean data
chat_file = "../data/whatsapp_chat2.txt"  # Replace with your chat file
parser = Parser(chat_file)
df = utils.df_basic_cleanup(parser.parse_chat_data())

# Define custom Hinglish stopwords
custom_hinglish_stopwords = set([
    '<media omitted>', 'media', 'omitted', 'bhai', 'hai', 'kya', 'ka', 'ki', 'ke', 'h', 'nahi', 'haan', 'ha',
    'to', 'ye', 'ho', 'na', 'ko', 'se', 'me', 'mai', 'mera', 'apna', 'tum', 'mujhe', 'jo',
    'bhi', 'nhi', 'hi', 'rha', 'tha', 'hain', 'abhi', 'kr', 'rha', 'thi', 'kar', 'karna',
    'raha', 'rahe', 'gaya', 'gayi', 'kyun', 'acha', 'lo', 'pe', 'kaun', 'tumhare', 'unki',
    'message', 'wo', 'koi', 'aa', 'le', 'ek', 'mei', 'lab', 'aur', 'kal', 'sab', 'us', 'un',
    'hum', 'kab', 'ab', 'par', 'kaise', 'unka', 'ap', 'mere', 'tere', 'kar', 'deleted', 'hun', 'hu', 'ne',
    'tu', 'ya', 'edited'
])

skill_keywords = {
    'communication': [
        'talk', 'discuss', 'share', 'convey', 'express', 'message', 'articulate',
        'explain', 'correspond', 'batana', 'samjhana', 'bataana', 'baat', 'dono',
        'tell', 'suno', 'dikhana', 'bol', 'bolna', 'likhna', 'likh', 'samaj',
        'sun', 'keh', 'kehna', 'padhana', 'janana', 'jan', 'vyakth karna', 'samjhao',
        'dekh', 'dekhna','sunana','samvad','guftgu','prastut','izhaar','pragatikaran','viniyog'
    ],
    'leadership': [
        'guide', 'manage', 'lead', 'organize', 'direct', 'influence', 'motivate',
        'inspire', 'leadership', 'rahnumai', 'neta banna', 'lead karna', 'manage karna',
        'prabhaavit karna', 'dhikhaana', 'aguvai', 'nirdeshan', 'niyantran',
        'prabandhak', 'netritvakarta', 'pravartak', 'diksha', 'dekhrekh','chalana','niyantran karna'
    ],
    'problem_solving': [
        'solve', 'resolve', 'analyze', 'figure', 'fix', 'improve', 'optimize',
        'address', 'determine', 'solve karna', 'masla suljhna', 'improve karna',
        'sahi karna', 'thik karna', 'dhoondhna', 'hal karna', 'samadhan', 'niptara',
        'sudharna', 'behtar', 'anukulan', 'nirdharan',  'gyat','thik karna',
        'samadhan sochna', 'samadhan ka upyog', 'samadhanikaran', 'samadhan dena'
    ],
    'technical': [
        'code', 'program', 'algorithm', 'software', 'hardware', 'system', 'network',
        'database', 'debug', 'coding', 'programming', 'debugging', 'networking',
        'computer', 'server', 'database kaam', 'tech', 'cloud', 'app', 'automation',
        'hardware ki setting', 'takniki', 'praudyogiki', 'yantrik', 'abhikalpan',
        'karya', 'karya pranali', 'vidhi', 'tantra','upkaran', 'samagri', 'sangathan', 
        'sanchar', 'aankda', 'soochi', 'doshal', 'tantrik', 'vigyan', 'software vikas',
        'hardware vikas', 'network sthapana', 'database prabandhan', 'debug karna'
    ],
    'teamwork': [
        'collaborate', 'cooperate', 'coordinate', 'assist', 'support', 'together',
        'contribute', 'participate', 'teamwork', 'saath kaam karna', 'mil jul kar kaam',
        'sath dena', 'madad karna', 'sahyog karna', 'support karna', 'cooperate karna',
        'milkar', 'sath', 'sahkarya', 'sajha', 'sahkari', 'sahbhaagi', 'samudaayik', 'ekjut',
        'sammilit', 'gatbandhan','sahyog dena'
    ]
}


# Combine NLTK stopwords with custom Hinglish stopwords
stop_words = set(stopwords.words('english')).union(custom_hinglish_stopwords)

@lru_cache(maxsize=None)  # Cache all unique calls
def clean_message(msg):
    """
    Clean the message by removing URLs, media omitted phrases, and trimming spaces.
    """
    # Remove URLs
    msg = re.sub(r'http[s]?://\S+', '', msg)
    # Remove "media omitted" phrases, case-insensitive
    msg = re.sub(r'\b(media omitted|<media omitted>)\b', '', msg, flags=re.IGNORECASE)
    # Strip any extra spaces
    msg = msg.strip()
    return msg

def extract_emojis(text):
    """Extract emojis from text."""
    return [c for c in text if c in emoji.EMOJI_DATA]

def plot_to_base64(plt):
    """Convert a Matplotlib plot to a base64 encoded image."""
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def apply_consistent_plot_styling(plt, title, xlabel, ylabel):
    """Applies consistent styling to Matplotlib plots."""
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.tight_layout()

def plot_activity_heatmap(df, username=None):
    """Plot an activity heatmap and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered['weekday'] = df_filtered['date'].dt.day_name()
    df_filtered['hour'] = df_filtered['hour']

    heatmap_data = df_filtered.pivot_table(index='weekday', columns='hour', values='message', aggfunc='count', fill_value=0)
    heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    plt.figure(figsize=(12, 6), constrained_layout=True)
    sns.heatmap(heatmap_data, cmap='viridis', annot=False, cbar_kws={'label': 'Number of Messages'})
    apply_consistent_plot_styling(plt, f'Activity Heatmap {"for " + username if username else ""}', 'Hour of the Day', 'Day of the Week')
    return plot_to_base64(plt)


def plot_sentiment_distribution(df, username=None):
    """Plot sentiment distribution and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    plt.figure(figsize=(8, 5))
    sns.histplot(df_filtered['sentiment'], bins=20, kde=True, color='skyblue')
    apply_consistent_plot_styling(plt, f'Sentiment Distribution {"for " + username if username else ""}', 'Sentiment Polarity', 'Frequency')
    return plot_to_base64(plt)

def plot_most_active_hours(df, username=None):
    """Plot a bar chart of the most active hours and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    message_counts_by_hour = df_filtered['hour'].value_counts().sort_index()

    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.bar(message_counts_by_hour.index, message_counts_by_hour.values, color='skyblue')
    apply_consistent_plot_styling(plt, f'Most Active Hours {"for " + username if username else ""}', 'Hour of the Day', 'Number of Messages')
    return plot_to_base64(plt)

def generate_wordcloud(df, username=None):
    """Generate word cloud and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    text = " ".join(msg for msg in df_filtered['clean_message'] if isinstance(msg, str) and len(msg)>1)

    wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f'Word Cloud {"for " + username if username else ""}', fontsize=14)
    #plt.tight_layout()
    return plot_to_base64(plt)

def analyze_language_complexity(df, username=None):
    """Analyze language complexity and return base64 images."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))
    
    # filter out emojis
    df_filtered['word_length'] = df_filtered['clean_message'].apply(
        lambda x: [len(word) for word in str(x).split() if word.lower() not in stop_words and len(word) > 1 and not all(c in emoji.EMOJI_DATA for c in word)]
    )
    
    avg_word_lengths = df_filtered['word_length'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

    # Handle cases with only emojis or empty messages
    df_filtered['sentence_length'] = df_filtered['clean_message'].apply(
        lambda x: len(nltk.sent_tokenize(str(x))) if str(x).strip() else 0
    )
    avg_sentence_lengths = df_filtered['sentence_length'].apply(
        lambda x: len(str(x).split()) / x if x > 0 and len(str(x).split()) > 0 else 0
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(avg_word_lengths, bins=20, kde=True, color='skyblue', ax=axs[0])
    apply_consistent_plot_styling(plt, f'Average Word Length {"for " + username if username else ""}', 'Average Word Length', 'Frequency')

    sns.histplot(avg_sentence_lengths, bins=20, kde=True, color='salmon', ax=axs[1])
    apply_consistent_plot_styling(plt, f'Average Sentence Length {"for " + username if username else ""}', 'Average Sentence Length (words)', 'Frequency')

    #plt.tight_layout()
    
    # Convert the combined plot to base64
    combined_plot_base64 = plot_to_base64(plt)
    
    return combined_plot_base64

def analyze_message_timing(df, username=None):
    """Analyze the timing of messages and return response times."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['time_diff'] = df_filtered.groupby('name')['date'].diff()
    response_times = df_filtered['time_diff'].dropna().apply(lambda x: x.total_seconds() / 60)  # in minutes

    return response_times

def plot_response_time_distribution(response_times, username=None):
    """Plot the distribution of response times."""
    plt.figure(figsize=(8, 5))
    sns.histplot(response_times, bins=20, kde=True, color='skyblue')
    apply_consistent_plot_styling(plt, f'Response Time Distribution {"for " + username if username else ""}', 'Response Time (minutes)', 'Frequency')
    return plot_to_base64(plt)

def analyze_sentiment_over_time(df, username=None):
    """Analyze sentiment over time and return base64 image of the plot."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    # Resample to daily frequency and calculate the mean sentiment
    daily_sentiment = df_filtered['sentiment'].resample('D').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment.index, daily_sentiment.values, color='purple')
    apply_consistent_plot_styling(plt, f'Sentiment Over Time {"for " + username if username else ""}', 'Date', 'Average Sentiment')
    
    return plot_to_base64(plt)

def analyze_emotion_over_time(df, username=None):
    """Analyze emotion over time using TextBlob and return base64 image of the plot."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()
    
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered.set_index('date', inplace=True)

    # Define a function to categorize sentiment into emotions
    def categorize_emotion(score):
        if score > 0.5:
            return "joy"
        elif score > 0:
            return "surprise"
        elif score < -0.5:
            return "sadness"
        elif score < 0:
            return "anger"
        else:
            return "neutral"
    
    # Apply sentiment analysis and emotion categorization
    df_filtered['sentiment'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_filtered['emotion'] = df_filtered['sentiment'].apply(categorize_emotion)
    
    # Resample to daily frequency and count the occurrences of each emotion
    daily_emotions = df_filtered.groupby(pd.Grouper(freq='D'))['emotion'].apply(lambda x: x.value_counts()).unstack(fill_value=0)
    
    
    plt.figure(figsize=(12, 6))
    for emotion in daily_emotions.columns:
        plt.plot(daily_emotions.index, daily_emotions[emotion], label=emotion)
    plt.legend()
    apply_consistent_plot_styling(plt, f'Emotion Trends Over Time {"for " + username if username else ""}', 'Date', 'Emotion Score')
    
    return plot_to_base64(plt)

def plot_emoji_usage(df, username=None):
    """Plot a bar chart of the top 5 emojis used and return base64 image."""
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    df_filtered['emojis'] = df_filtered['message'].apply(extract_emojis)
    all_emojis = [emoji for sublist in df_filtered['emojis'] for emoji in sublist]
    top_emojis = Counter(all_emojis).most_common(5)

    emojis, counts = zip(*top_emojis)
    plt.figure(figsize=(10, 6))
    plt.bar(emojis, counts, color='skyblue')
    apply_consistent_plot_styling(plt, f'Emoji Usage {"for " + username if username else ""}', 'Emoji', 'Count')
    return plot_to_base64(plt)

def plot_sentiment_bubble(df, username=None):
    """
    Plot a bubble chart of sentiment distribution and return base64 image.
    x-axis: Polarity (Positive/Negative)
    y-axis: Subjectivity (Objective/Subjective)
    Bubble size: Number of messages
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Calculate sentiment polarity and subjectivity
    df_filtered['polarity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_filtered['subjectivity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    # Count the number of messages for each sentiment
    sentiment_counts = df_filtered.groupby(['polarity', 'subjectivity']).size().reset_index(name='counts')

    plt.figure(figsize=(10, 8))
    plt.scatter(sentiment_counts['polarity'], sentiment_counts['subjectivity'], s=sentiment_counts['counts']*10, alpha=0.6, color='purple')
    apply_consistent_plot_styling(plt, f'Sentiment Distribution {"for " + username if username else ""}', 'Polarity (Positive/Negative)', 'Subjectivity (Objective/Subjective)')
    return plot_to_base64(plt)

def plot_vocabulary_diversity(df, username=None):
    """
    Plot a scatter plot of vocabulary diversity over time and return base64 image.
    x-axis: Unique words used
    y-axis: Average message length
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()
        
    df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower() # optimization for unique words
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    word_matrix = vectorizer.fit_transform(df_filtered['clean_message_lower'].dropna())
    unique_words_count = len(vectorizer.get_feature_names_out())

    # Calculate average message length
    df_filtered['avg_message_length'] = df_filtered['message'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 8))
    plt.scatter(unique_words_count, df_filtered['avg_message_length'].mean(), color='green')
    apply_consistent_plot_styling(plt, f'Vocabulary Diversity {"for " + username if username else ""}', 'Unique Words (Avg)', 'Message Length (Avg)')
    return plot_to_base64(plt)

def plot_language_complexity_pos(df, username=None):
    """
    Analyze and plot the distribution of POS tags for a user or the entire chat,
    and return a base64 image of the plot.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Function to extract POS tags from a message
    def extract_pos_tags(message):
        analysis = TextBlob(message)
        return [tag for (word, tag) in analysis.tags]

    # Apply POS tag extraction to each message
    df_filtered['pos_tags'] = df_filtered['message'].apply(extract_pos_tags)

    # Flatten the list of POS tags and count their occurrences
    all_pos_tags = [tag for sublist in df_filtered['pos_tags'] for tag in sublist]
    pos_counts = Counter(all_pos_tags)

    # Convert to DataFrame for plotting
    pos_df = pd.DataFrame(list(pos_counts.items()), columns=['POS Tag', 'Count'])

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='POS Tag', y='Count', data=pos_df, color='skyblue')
    apply_consistent_plot_styling(plt, f'POS Tag Distribution {"for " + username if username else ""}', 'POS Tag', 'Count')

    return plot_to_base64(plt)

def plot_user_relationship_graph(df):
    """
    Plot a graph representing the relationships between users based on message interactions.
    Nodes represent users, and edges represent interactions between them.
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes for each user
    for user in df['name'].unique():
        G.add_node(user)

    # Analyze interactions and add edges
    for i in range(len(df) - 1):
        sender = df['name'].iloc[i]
        next_sender = df['name'].iloc[i + 1]
        if sender != next_sender:
            if G.has_edge(sender, next_sender):
                G[sender][next_sender]['weight'] += 1
            else:
                G.add_edge(sender, next_sender, weight=1)

    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=8, width=[d['weight'] / 5 for u, v, d in G.edges(data=True)])
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("User Relationship Graph", fontsize=15)
    #plt.tight_layout()

    return plot_to_base64(plt)

def analyze_behavioral_traits(df, username=None):
    """
    Analyze behavioral traits and return a dictionary of insights.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Placeholder for more advanced analysis
    traits = {}

    # Example: Check for question-asking tendency
    question_count = df_filtered['message'].apply(lambda x: x.count('?')).sum()
    total_messages = len(df_filtered)
    traits['question_tendency'] = question_count / total_messages if total_messages > 0 else 0

    # Example: Check for use of long sentences (which might indicate detailed explanations)
    df_filtered['sentence_length'] = df_filtered['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))))
    avg_sentence_length = df_filtered['sentence_length'].mean()
    traits['avg_sentence_length'] = avg_sentence_length

    # Add more traits as needed based on your specific analysis goals

    return traits

def generate_behavioral_insights_text(traits, most_active_period, avg_response_time):
    """
    Generate human-readable insights based on behavioral traits.
    """
    insights = []

    if traits['question_tendency'] > 0.2:  # Example threshold
        insights.append("Asks a lot of questions, possibly indicating curiosity or a need for clarification.")
    else:
        insights.append("Rarely asks questions, possibly indicating self-sufficiency or confidence in the subject matter.")

    if traits['avg_sentence_length'] > 3:  # Example threshold
        insights.append("Uses long and complex sentences.")
    else:
        insights.append("Uses short and concise sentences.")

    # Add more insights based on other traits you've analyzed

    return "\n".join(insights)

def plot_skills_radar_chart(df, username=None):
    """
    Generate a radar chart to visualize various skills based on keyword analysis.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    

    # Count keyword occurrences for each skill
    skill_counts = {}
    for skill, keywords in skill_keywords.items():
        skill_counts[skill] = sum(df_filtered['clean_message'].str.lower().str.count('|'.join(keywords)))

    # Prepare data for radar chart
    skills = list(skill_counts.keys())
    counts = list(skill_counts.values())
    
    # Normalize counts for radar chart
    max_count = max(counts)
    normalized_counts = [(count / max_count) for count in counts]

    # Number of variables (skills)
    num_vars = len(skills)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Initialize radar chart
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], skills, color='black', size=10)

    # Draw ylabels (normalized counts)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, normalized_counts + normalized_counts[:1], linewidth=2, linestyle='solid')
    ax.fill(angles, normalized_counts + normalized_counts[:1], 'b', alpha=0.1)

    # Add title
    plt.title(f'Skills Radar Chart {"for " + username if username else ""}', size=15, color='black', y=1.1)
    #plt.tight_layout()

    return plot_to_base64(plt)
def analyze_behavioral_traits(df, username=None):
    """
    Analyze behavioral traits and return a dictionary of insights.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    traits = {}

    # --- Sentiment Analysis ---
    df_filtered['sentiment_polarity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_filtered['sentiment_subjectivity'] = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    traits['avg_sentiment_polarity'] = df_filtered['sentiment_polarity'].mean()
    traits['avg_sentiment_subjectivity'] = df_filtered['sentiment_subjectivity'].mean()

    # --- Psychometric Analysis ---
    traits['num_questions'] = df_filtered['message'].apply(lambda x: x.count('?')).sum()
    traits['num_exclamations'] = df_filtered['message'].apply(lambda x: x.count('!')).sum()
    traits['first_person_pronouns'] = df_filtered['clean_message'].str.lower().str.count(r'\b(i|me|my|mine|myself)\b').sum() 

    # --- Skill Analysis (Keyword-based) ---
    
    traits['skills'] = {}
    for skill, keywords in skill_keywords.items():
        traits['skills'][skill] = sum(df_filtered['clean_message'].str.lower().str.count('|'.join(keywords)))

    # # --- Timing Analysis ---
    # if 'avg_response_time' in basic_stats:
    #     traits['avg_response_time'] = user_stats['Average Response Time']
    
    # --- Language Complexity ---
    df_filtered['sentence_length'] = df_filtered['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))) if str(x).strip() else 0)
    traits['avg_sentence_length'] = df_filtered['sentence_length'].mean()

    # --- Lexical Diversity ---
    df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower()
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    word_matrix = vectorizer.fit_transform(df_filtered['clean_message_lower'].dropna())
    unique_words_count = len(vectorizer.get_feature_names_out())
    total_words_count = df_filtered['message'].apply(lambda x: len(str(x).split())).sum()
    traits['lexical_diversity'] = unique_words_count / total_words_count if total_words_count > 0 else 0

    return traits

def generate_behavioral_insights_text(traits, most_active_period, avg_response_time):
    """
    Generate human-readable insights based on behavioral traits.
    """
    insights = []

    # Sentiment Hints
    if traits['avg_sentiment_polarity'] > 0.2:
        insights.append("Tends to express positive sentiment in messages.")
    elif traits['avg_sentiment_polarity'] < -0.2:
        insights.append("Tends to express negative sentiment in messages.")
    else:
        insights.append("Maintains a neutral tone in messages.")

    if traits['avg_sentiment_subjectivity'] > 0.5:
        insights.append("Expresses subjective opinions and evaluations.")
    else:
        insights.append("Tends to communicate more objectively.")

    # Psychometric Hints
    if traits['num_questions'] > 20:
        insights.append("Asks a lot of questions, possibly indicating curiosity or a need for clarification.")
    if traits['num_exclamations'] > 5:
        insights.append("Uses exclamations frequently, suggesting excitement or strong opinions.")
    if traits['first_person_pronouns'] > 10:
        insights.append("Often refers to themselves, which might indicate a focus on personal experiences or opinions.")

    # Skill Hints
    if traits['skills']['communication'] > 5:
        insights.append("Demonstrates strong communication skills based on keyword analysis.")
    if traits['skills']['technical'] > 5:
        insights.append("Exhibits technical skills based on keyword analysis.")
    if traits['skills']['leadership'] > 2:
        insights.append("Shows potential leadership qualities based on keyword analysis.")
    if traits['skills']['problem_solving'] > 5:
        insights.append("Appears to have good problem-solving skills based on keyword analysis.")
    if traits['skills']['teamwork'] > 5:
        insights.append("Likely a good team player based on keyword analysis.")

    # Timing Hints
    if avg_response_time is not None:
        if avg_response_time < 60:
            insights.append("Responds quickly to messages, indicating high engagement.")
        elif avg_response_time > 180:
            insights.append("Takes longer to respond, suggesting lower engagement or a busy schedule.")
        else:
            insights.append("Has a moderate response time.")

    if most_active_period is not None:
        if most_active_period == 'Morning':
            insights.append("Most active in the morning.")
        elif most_active_period == 'Mid-day':
            insights.append("Most active in the afternoon.")
        elif most_active_period == 'Evening':
            insights.append("Most active in the evening.")
        else:
            insights.append("Most active at night.")
    
    # Language Complexity Hints
    if traits['avg_sentence_length'] > 3:
        insights.append("Uses long and complex sentences.")
    else:
        insights.append("Uses short and concise sentences.")

    # Lexical Diversity Hints
    if traits['lexical_diversity'] > 0.7:
        insights.append("Exhibits high lexical diversity, indicating a broad vocabulary.")
    elif traits['lexical_diversity'] < 0.4:
        insights.append("Has low lexical diversity, suggesting a more repetitive or focused communication style.")
    else:
        insights.append("Shows moderate lexical diversity.")

    return "<br/>".join(insights)

def analyze_hindi_abuse(df, username=None):
    """
    Analyze the use of Hindi abusive words and return a dictionary of counts.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # List of Hindi abusive words 
    hindi_abusive_words = ["<fill words>"]


    # Count occurrences of each abusive word
    abuse_counts = {}
    for word in hindi_abusive_words:
        count = df_filtered['clean_message'].str.lower().str.count(word).sum()
        if count > 1:  # Only include if count is greater than 1
            abuse_counts[word] = count

    return abuse_counts

def basic_stats(df, username=None):
    """
    Calculate basic statistics about messages, including sentiment, time analysis,
    most common n-grams (unigrams, bigrams, trigrams), most active period, and visualizations.
    """
    if username:
        df_filtered = df[df['name'] == username].copy()
    else:
        df_filtered = df.copy()

    # Sentiment Analysis
    sentiments = df_filtered['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    positive_msgs = sum(sentiments > 0)
    negative_msgs = sum(sentiments < 0)

    # Time of Day Analysis
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 16:
            return 'Mid-day'
        elif 16 <= hour < 18:
            return 'Evening'
        else:
            return 'Night'

    df_filtered['time_of_day'] = df_filtered['hour'].apply(categorize_time_of_day)
    morning_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Morning'])
    midday_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Mid-day'])
    evening_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Evening'])
    night_msgs = len(df_filtered[df_filtered['time_of_day'] == 'Night'])
    message_counts_by_period = {'Morning': morning_msgs, 'Mid-day': midday_msgs, 'Evening': evening_msgs, 'Night': night_msgs}
    most_active_period = max(message_counts_by_period, key=message_counts_by_period.get)

    # Clean messages
    df_filtered['clean_message'] = df_filtered['message'].apply(lambda x: clean_message(str(x)))

    # Unique words count (optimized)
    df_filtered['clean_message_lower'] = df_filtered['clean_message'].str.lower() # optimization for unique words
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    word_matrix = vectorizer.fit_transform(df_filtered['clean_message_lower'].dropna())
    unique_words_count = len(vectorizer.get_feature_names_out())

    # Most Common unigrams, bigrams, trigrams (optimized)
    def get_top_ngrams(corpus, n=1, top_k=10):
        vec = CountVectorizer(ngram_range=(n, n), stop_words=list(stop_words)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq_filtered = [item for item in words_freq if item[1] > 1]
        words_freq = sorted(words_freq_filtered, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    
    common_unigrams = get_top_ngrams(df_filtered['clean_message_lower'].dropna(), 1, 10)
    common_bigrams = get_top_ngrams(df_filtered['clean_message_lower'].dropna(), 2, 10)
    common_trigrams = get_top_ngrams(df_filtered['clean_message_lower'].dropna(), 3, 10)

    # Top 5 Emojis
    df_filtered['emojis'] = df_filtered['message'].apply(extract_emojis)
    all_emojis = [emoji for sublist in df_filtered['emojis'] for emoji in sublist]
    top_5_emojis = Counter(all_emojis).most_common(5)

    # Average Sentence Length
    df_filtered['sentence_length'] = df_filtered['clean_message'].apply(lambda x: len(nltk.sent_tokenize(str(x))))
    avg_sentence_length = df_filtered['sentence_length'].apply(lambda x: len(str(x).split()) / x if x > 0 else 0).mean()

    # Analyze message timing and get response times
    response_times = analyze_message_timing(df, username)
    
    # Calculate average response time
    average_response_time = response_times.mean() if not response_times.empty else 0

    # Visualizations
    activity_heatmap_base64 = plot_activity_heatmap(df_filtered, username)
    sentiment_distribution_base64 = plot_sentiment_distribution(df_filtered, username)
    wordcloud_base64 = generate_wordcloud(df_filtered, username)
    language_complexity_base64 = analyze_language_complexity(df_filtered, username)
    response_time_distribution_base64 = plot_response_time_distribution(response_times, username)
    sentiment_over_time_base64 = analyze_sentiment_over_time(df, username)
    emoji_usage_base64 = plot_emoji_usage(df_filtered, username)
    sentiment_bubble_base64 = plot_sentiment_bubble(df_filtered, username)
    vocabulary_diversity_base64 = plot_vocabulary_diversity(df_filtered, username)
    language_complexity_pos_base64 = plot_language_complexity_pos(df_filtered, username)
    user_relationship_graph_base64 = plot_user_relationship_graph(df)
    skills_radar_chart_base64 = plot_skills_radar_chart(df_filtered, username)
    emotion_over_time_base64 = analyze_emotion_over_time(df_filtered, username)
    most_active_hours_base64 = plot_most_active_hours(df_filtered, username)
    # Analyze behavioral traits
    behavioral_traits = analyze_behavioral_traits(df_filtered, username)
    behavioral_insights_text = generate_behavioral_insights_text(behavioral_traits, most_active_period, average_response_time)

    # Analyze for Hindi गाली and get counts if count > 1
    abuse_counts = analyze_hindi_abuse(df_filtered, username)
    
    # Convert the abuse_counts dictionary to an HTML-formatted string
    abuse_counts_html = "<ul>"
    for word, count in abuse_counts.items():
        abuse_counts_html += f"<li>{word}: {count}</li>"
    abuse_counts_html += "</ul>"

    stats = {
        'Total Messages': len(df_filtered),
        'Total Words': df_filtered['message'].apply(lambda x: len(str(x).split())).sum(),
        'Unique Users': df_filtered['name'].nunique(),
        'Total Emojis': df_filtered['emojicount'].sum(),
        'Total URLs': df_filtered['urlcount'].sum(),
        'Total YouTube URLs': df_filtered['yturlcount'].sum(),
        'Total Media': df_filtered['mediacount'].sum(),
        'Total Edits': df_filtered['editcount'].sum(),
        'Total Deletes': df_filtered['deletecount'].sum(),
        'Average Message Length': df_filtered['message'].apply(lambda x: len(str(x).split())).mean(),
        'Positive Messages': positive_msgs,
        'Negative Messages': negative_msgs,
        'Morning Messages': morning_msgs,
        'Mid-day Messages': midday_msgs,
        'Evening Messages': evening_msgs,
        'Night Messages': night_msgs,
        'Most Active Period': most_active_period,
        'Unique Words Count': unique_words_count,
        'Common Unigrams': common_unigrams,
        'Common Bigrams': common_bigrams,
        'Common Trigrams': common_trigrams,
        'Top 5 Emojis': top_5_emojis,
        'Average Sentence Length': avg_sentence_length,
        'Average Response Time': average_response_time,
        'Activity Heatmap': activity_heatmap_base64,
        'Sentiment Distribution': sentiment_distribution_base64,
        'Word Cloud': wordcloud_base64,
        'Language Complexity': language_complexity_base64,
        'Response Time Distribution': response_time_distribution_base64,
        'Sentiment Over Time': sentiment_over_time_base64,
        'Emoji Usage': emoji_usage_base64,
        'Sentiment Bubble': sentiment_bubble_base64,
        'Vocabulary Diversity': vocabulary_diversity_base64,
        'Language Complexity POS': language_complexity_pos_base64,
        'User Relationship Graph': user_relationship_graph_base64,
        'Skills Radar Chart': skills_radar_chart_base64,
        'Behavioral Traits': behavioral_traits,
        'Emotion Over Time': emotion_over_time_base64,
        'Behavioral Insights Text': behavioral_insights_text,
        'Hindi Abuse Counts': abuse_counts_html,
        'Most Active Hours': most_active_hours_base64,
    }

    return stats

# Enhanced HTML Template and Styling
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhatsApp Chat Analysis - {name}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>WhatsApp Chat Analysis - {name}</h1>
        </header>
        <div class="row mt-4">
            <div class="col-md-3">
                <div class="profile-card">
                    <img src="https://via.placeholder.com/150" alt="{name}'s Profile Picture" class="profile-img">
                    <h3 class="username">{name}</h3>
                    <p class="status">Active User</p>
                    <p class="location"><i class="fas fa-map-marker-alt"></i> Location: New Delhi</p>
                    
                </div>
            </div>
            <div class="col-md-9">
                <div class="user-report">
                    <div class="section">
                        <h2 class="section-title">User Stats</h2>
                        <table class="table table-striped">
                            <tr><th>Total Messages</th><td>{total_messages}</td></tr>
                            <tr><th>Total Words</th><td>{total_words}</td></tr>
                            <tr><th>Unique Users</th><td>{unique_users}</td></tr>
                            <tr><th>Total Emojis</th><td>{total_emojis}</td></tr>
                            <tr><th>Top 5 Emojis</th><td class="emoji">{top_5_emojis}</td></tr>
                            <tr><th>Total URLs</th><td>{total_urls}</td></tr>
                            <tr><th>Total YouTube URLs</th><td>{total_youtube_urls}</td></tr>
                            <tr><th>Total Media</th><td>{total_media}</td></tr>
                            <tr><th>Total Edits</th><td>{total_edits}</td></tr>
                            <tr><th>Total Deletes</th><td>{total_deletes}</td></tr>
                            <tr><th>Average Message Length</th><td>{average_message_length:.2f}</td></tr>
                            <tr><th>Average Sentence Length</th><td>{average_sentence_length:.2f}</td></tr>
                            <tr><th>Positive Messages</th><td>{positive_messages}</td></tr>
                            <tr><th>Negative Messages</th><td>{negative_messages}</td></tr>
                            <tr><th>Morning Messages</th><td>{morning_messages}</td></tr>
                            <tr><th>Mid-day Messages</th><td>{midday_messages}</td></tr>
                            <tr><th>Evening Messages</th><td>{evening_messages}</td></tr>
                            <tr><th>Night Messages</th><td>{night_messages}</td></tr>
                            <tr><th>Most Active Period</th><td>{most_active_period}</td></tr>
                            <tr><th>Unique Words Count</th><td>{unique_words_count}</td></tr>
                            <tr><th>Average Response Time (minutes)</th><td>{average_response_time:.2f}</td></tr>
                        </table>
                    </div>
                    <div class="section">
                        <h2 class="section-title">Common Words</h2>
                        <h3>Unigrams</h3>
                        <ul>
                            {common_unigrams}
                        </ul>
                        <h3>Bigrams</h3>
                        <ul>
                            {common_bigrams}
                        </ul>
                        <h3>Trigrams</h3>
                        <ul>
                            {common_trigrams}
                        </ul>
                        <h3>Hindi abuse</h3>
                        <ul>
                            {hindi_abuse_count}
                        </ul>
                    </div>
                    <div class="section">
                        <h2 class="section-title">Visualizations</h2>
                              
                        <div class="visualization">
                            <h4>Most Active Hours</h4>
                            <img src="data:image/png;base64,{most_active_hours}" alt="Most Active Hours">
                        </div>

    
                        <div class="visualization">
                            <h4>Activity Heatmap</h4>
                            <img src="data:image/png;base64,{activity_heatmap}" alt="Activity Heatmap">
                        </div>
                        <div class="visualization">
                            <h4>Response Time Distribution</h4>
                            <img src="data:image/png;base64,{response_time_distribution}" alt="Response Time Distribution">
                        </div>
                        <div class="visualization">
                            <h4>Sentiment Over Time</h4>
                            <img src="data:image/png;base64,{sentiment_over_time}" alt="Sentiment Over Time">
                        </div>
                        <div class="visualization">
                            <h4>Emoji Usage</h4>
                            <img src="data:image/png;base64,{emoji_usage}" alt="Emoji Usage">
                        </div>
                        <div class="visualization">
                            <h4>Sentiment Distribution</h4>
                            <img src="data:image/png;base64,{sentiment_distribution}" alt="Sentiment Distribution">
                        </div>
                        <div class="visualization">
                            <h4>Sentiment (Bubble)</h4>
                            <img src="data:image/png;base64,{sentiment_bubble}" alt="Sentiment Bubble">
                        </div>
                        <div class="visualization">
                            <h4>Vocabulary Diversity</h4>
                            <img src="data:image/png;base64,{vocabulary_diversity}" alt="Vocabulary Diversity">
                        </div>
                        <div class="visualization">
                            <h4>Language Complexity</h4>
                            <img src="data:image/png;base64,{language_complexity}" alt="Language Complexity">
                        </div>
                        <div class="visualization">
                            <h4>Language Complexity (POS)</h4>
                            <img src="data:image/png;base64,{language_complexity_pos}" alt="Language Complexity POS">
                        </div>
                        <div class="visualization">
                            <h4>User Relationship Graph</h4>
                            <img src="data:image/png;base64,{user_relationship_graph}" alt="User Relationship Graph">
                        </div>
                        <div class="visualization">
                            <h4>Skills Radar Chart</h4>
                            <img src="data:image/png;base64,{skills_radar_chart}" alt="Skills Radar Chart">
                        </div>
                        <div class="visualization">
                            <h4>Emotion Trends (Time Series)</h4>
                            <img src="data:image/png;base64,{emotion_over_time}" alt="Emotion Over Time">
                        </div>
                        <div class="visualization">
                            <h4>Word Cloud</h4>
                            <img src="data:image/png;base64,{word_cloud}" alt="Word Cloud">
                        </div>
                    </div>
                    <div class="section">
                        <h2 class="section-title">Behavioral Insights</h2>
                        <div class="insights">
                            {behavioral_insights_text}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <footer class="footer">
            <p>Generated with <i class="fas fa-heart"></i> by WhatsApp Analyzer</p>
            <p><a href="https://github.com/your-repo-link" target="_blank"><i class="fab fa-github"></i> Visit the Project</a></p>
        </footer>
    </div>
</body>
</html>
"""

# Generate HTML report for each user
for name in df['name'].unique():
    user_stats = basic_stats(df, name)
    top_5_emojis_html = " ".join([f"{emoji} ({count})" for emoji, count in user_stats['Top 5 Emojis']])

    final_html = html_template.format(
        name=name,
        total_messages=user_stats['Total Messages'],
        total_words=user_stats['Total Words'],
        unique_users=user_stats['Unique Users'],
        total_emojis=user_stats['Total Emojis'],
        top_5_emojis=top_5_emojis_html,
        total_urls=user_stats['Total URLs'],
        total_youtube_urls=user_stats['Total YouTube URLs'],
        total_media=user_stats['Total Media'],
        total_edits=user_stats['Total Edits'],
        total_deletes=user_stats['Total Deletes'],
        average_message_length=user_stats['Average Message Length'],
        average_sentence_length=user_stats['Average Sentence Length'],
        positive_messages=user_stats['Positive Messages'],
        negative_messages=user_stats['Negative Messages'],
        morning_messages=user_stats['Morning Messages'],
        midday_messages=user_stats['Mid-day Messages'],
        evening_messages=user_stats['Evening Messages'],
        night_messages=user_stats['Night Messages'],
        most_active_period=user_stats['Most Active Period'],
        unique_words_count=user_stats['Unique Words Count'],
        common_unigrams="".join([f"<li>{word[0]}: {word[1]}</li>" for word in user_stats['Common Unigrams']]),
        common_bigrams="".join([f"<li>{word[0]}: {word[1]}</li>" for word in user_stats['Common Bigrams']]),
        common_trigrams="".join([f"<li>{word[0]}: {word[1]}</li>" for word in user_stats['Common Trigrams']]),
        average_response_time=user_stats['Average Response Time'],
        activity_heatmap=user_stats['Activity Heatmap'],
        sentiment_distribution=user_stats['Sentiment Distribution'],
        word_cloud=user_stats['Word Cloud'],
        language_complexity=user_stats['Language Complexity'],
        response_time_distribution=user_stats['Response Time Distribution'],
        sentiment_over_time=user_stats['Sentiment Over Time'],
        emoji_usage=user_stats['Emoji Usage'],
        sentiment_bubble=user_stats['Sentiment Bubble'],
        vocabulary_diversity=user_stats['Vocabulary Diversity'],
        language_complexity_pos=user_stats['Language Complexity POS'],
        user_relationship_graph=user_stats['User Relationship Graph'],
        skills_radar_chart=user_stats['Skills Radar Chart'],
        behavioral_insights_text=user_stats['Behavioral Insights Text'],
        emotion_over_time=user_stats['Emotion Over Time'],
        hindi_abuse_count=user_stats['Hindi Abuse Counts'],
        most_active_hours=user_stats['Most Active Hours'],
    )

    output_path = os.path.join('reports', f"{name}_report.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(final_html)

    print(f"Report for {name} has been generated and saved at {output_path}")
