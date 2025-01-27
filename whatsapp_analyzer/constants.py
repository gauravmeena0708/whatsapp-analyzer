import pandas as pd
from nltk.corpus import stopwords

# Now add the variables at the end of the file
custom_hinglish_stopwords = set([
    '<media omitted>', 'media', 'omitted', 'bhai', 'hai', 'kya', 'ka', 'ki', 'ke', 'h', 'nahi', 'haan', 'ha',
    'to', 'ye', 'ho', 'na', 'ko', 'se', 'me', 'mai', 'mera', 'apna', 'tum', 'mujhe', 'jo',
    'bhi', 'nhi', 'hi', 'rha', 'tha', 'hain', 'abhi', 'kr', 'rha', 'thi', 'kar', 'karna',
    'raha', 'rahe', 'gaya', 'gayi', 'kyun', 'acha', 'lo', 'pe', 'kaun', 'tumhare', 'unki',
    'message', 'wo', 'koi', 'aa', 'le', 'ek', 'mei', 'lab', 'aur', 'kal', 'sab', 'us', 'un',
    'hum', 'kab', 'ab', 'par', 'kaise', 'unka', 'ap', 'mere', 'tere', 'kar', 'deleted', 'hun', 'hu', 'ne',
    'tu', 'ya', 'edited'
])

# Combine NLTK stopwords with custom Hinglish stopwords
stop_words = set(stopwords.words('english')).union(custom_hinglish_stopwords)

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
        'hardware vikas', 'network sthapana', 'database prabandhan', 'debug karna',
        "lower bound", "upper bound", "time complexity", "space complexity", "algorithm design",
        "estimation", "nvidia", "detection", "classification", "regression", "prediction",
    ],
    'teamwork': [
        'collaborate', 'cooperate', 'coordinate', 'assist', 'support', 'together',
        'contribute', 'participate', 'teamwork', 'saath kaam karna', 'mil jul kar kaam',
        'sath dena', 'madad karna', 'sahyog karna', 'support karna', 'cooperate karna',
        'milkar', 'sath', 'sahkarya', 'sajha', 'sahkari', 'sahbhaagi', 'samudaayik', 'ekjut',
        'sammilit', 'gatbandhan','sahyog dena', "bhardo", "kardo", "bhejdo"
    ]
}

hindi_abusive_words = [
        'chutiya', 'gandu', 'bhosdike', 'bhadwe', 'madarchod', 'behenchod', 'randi',
        'laude', 'chut', 'harami', 'kutta', 'kutiya', 'suar', 'hijra', 'gaand', 'tatte',
        'jhat', 'bhosdi', 'bhadwa', 'chinal', 'chakka', 'behen ke laude', 'maa ke laude',
        'baap ke laude', 'bhosdiwala', 'bhosdiwali', 'gandu ke aulad', 'gandi aulad',
        'harami aulad', 'gandu sala', 'chutiya sala', 'bhosdike sala', 'madarchod sala',
        'gandi maa ka', 'gandi maa ki', 'gandu maa ka', 'gandu maa ki', 'chutiya maa ka',
        'chutiya maa ki', 'madarchod maa ka', 'madarchod maa ki', 'madarchod bhai',
        'madarchod bahen', 'bhosdike bhai', 'bhosdike bahen', 'chutiya bhai', 'chutiya bahen',
        'gandu bhai', 'gandu bahen', 'harami bhai', 'harami bahen', 'bhadwe bhai', 'bhadwe bahen',
        'bsdiwala', 'iski maka', 'betichod', "gand", "bc", "mc", "madar", "bkl", "bkl", "bkl", "bkl",]

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
    <style>
    
body {{
            font-family: "Segoe UI Emoji", "Apple Color Emoji", 'Roboto', Arial, sans-serif; 
            background-color: #f4f4f4;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: auto;
            padding: 20px;
        }}
        .header {{
            background-color: #075e54;
            color: #fff;
            padding: 20px;
            text-align: center;
            border-radius: 10px 10px 0 0;
        }}
        .profile-card {{
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            padding: 20px;
        }}
        .profile-img {{
            border-radius: 50%;
            width: 120px;
            height: 120px;
            margin: 0 auto 15px;
            object-fit: cover;
        }}
        .username {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        .status {{
            font-size: 1.1rem;
            color: #4CAF50;
            margin-bottom: 15px;
        }}
        .location, .social-links {{
            font-size: 1rem;
            color: #555;
            margin-bottom: 15px;
        }}
        .social-links a {{
            margin: 0 10px;
            color: #fff;
            padding: 8px 15px;
            border-radius: 5px;
            text-decoration: none;
        }}
        .social-links a.facebook {{ background-color: #3b5998; }}
        .social-links a.instagram {{ background-color: #e4405f; }}
        .user-report {{
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-top: 20px;
        }}
        .section-title {{
            color: #075e54;
            font-size: 1.5rem;
            margin-bottom: 15px;
        }}
        .table {{
            background-color: #fff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }}
        .table th {{
            background-color: #075e54;
            color: #fff;
        }}
        .table th, .table td {{
            padding: 12px 15px;
            text-align: left;
        }}
        .footer {{
            background-color: #075e54;
            color: #fff;
            text-align: center;
            padding: 20px;
            margin-top: 20px;
            border-radius: 0 0 10px 10px;
        }}
        .footer a {{
            color: #fff;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
        .emoji {{
            font-size: 1.2rem;
        }}
        .visualization {{
            margin-top: 20px;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .insights {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 10px;
        }}
        .insights h4 {{
            color: #075e54;
            margin-bottom: 10px;
        }}
        .insights p {{
            font-size: 0.9rem;
            line-height: 1.5;
        }}
		.emoji {{
            font-family: 'Roboto', Arial, sans-serif; 
        }}
    </style>
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
            <p><a href="https://github.com/gauravmeena0708/k" target="_blank"><i class="fab fa-github"></i> Visit the Project</a></p>
        </footer>
    </div>
</body>
</html>
"""
