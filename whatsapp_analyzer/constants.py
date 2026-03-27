from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.corpus import stopwords
except ModuleNotFoundError:
    stopwords = None

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

# Combine NLTK stopwords with custom Hinglish stopwords, but keep imports usable
# even when NLTK corpora have not been downloaded yet.
try:
    _english_stop_words = set(stopwords.words('english')) if stopwords is not None else set(ENGLISH_STOP_WORDS)
except LookupError:
    _english_stop_words = set(ENGLISH_STOP_WORDS)

stop_words = _english_stop_words.union(custom_hinglish_stopwords)

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
        'bsdiwala', 'iski maka', 'betichod', "gand", "bc", "mc", "madar", "bkl",]

# Big Five Personality Weights
personality_weights = {
    'Openness': {
        'lexical_diversity': 0.5,
        'unique_words_count': 0.5
    },
    'Conscientiousness': {
        'avg_message_length': 0.4,
        'avg_sentence_length': 0.4,
        'total_words': 0.2
    },
    'Extraversion': {
        'total_messages': 0.4,
        'emoji_density': 0.3,
        'exclamation_ratio': 0.3
    },
    'Agreeableness': {
        'avg_sentiment_polarity': 0.6,
        'abuse_ratio': -0.4
    },
    'Neuroticism': {
        'avg_sentiment_polarity': -0.4,
        'first_person_ratio': 0.6
    }
}

group_roles = {
    'The Mediator': {
        'description': 'Maintains harmony in the group. High Agreeableness and frequent positive interactions.',
        'icon': 'fa-handshake'
    },
    'The Energizer': {
        'description': 'Brings life to the chat with high energy, emojis, and frequent participation.',
        'icon': 'fa-bolt'
    },
    'The Expert': {
        'description': 'Provides deep insights and technical knowledge. Uses structured language and specific keywords.',
        'icon': 'fa-brain'
    },
    'The Lurker': {
        'description': 'Observes more than they speak. Low message frequency but present during active periods.',
        'icon': 'fa-eye'
    },
    'The Social Butterfly': {
        'description': 'Interacts with everyone. High Extraversion and a wide web of connections.',
        'icon': 'fa-comments'
    },
    'The Night Owl': {
        'description': 'Most active when the world sleeps. High percentage of messages during late-night hours.',
        'icon': 'fa-moon'
    }
}

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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
    
body {{
            font-family: 'Roboto', Arial, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            padding: 20px;
        }}
        .header {{
            background-color: #075e54;
            color: #fff;
            padding: 20px 20px 18px;
            text-align: center;
            border-radius: 10px 10px 0 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .header-meta {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            flex-wrap: wrap;
            margin-bottom: 8px;
        }}
        .profile-img {{
            border-radius: 50%;
            width: 64px;
            height: 64px;
            margin: 0;
            object-fit: cover;
        }}
        .header-copy {{
            text-align: left;
            min-width: 0;
        }}
        .header-copy h1 {{
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 2px;
        }}
        .header-copy p {{
            margin-bottom: 0;
            opacity: 0.9;
        }}
        .status {{
            font-size: 0.98rem;
            color: #d7ffe8;
            font-weight: bold;
        }}
        .user-report {{
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 20px;
        }}
        .section-title {{
            color: #075e54;
            font-size: 1.6rem;
            margin-bottom: 20px;
            border-bottom: 2px solid #ececec;
            padding-bottom: 10px;
        }}
        .table {{
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }}
        .table th {{
            background-color: #075e54;
            color: #fff;
            font-weight: 500;
        }}
        .table th, .table td {{
            padding: 12px 15px;
            vertical-align: middle;
        }}
        .footer {{
            background-color: #075e54;
            color: #fff;
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            border-radius: 10px;
            box-shadow: 0 -4px 8px rgba(0,0,0,0.1);
        }}
        .footer a {{
            color: #fff;
            text-decoration: none;
            font-weight: bold;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
        .emoji {{
            font-size: 1.4rem;
            font-family: "Segoe UI Emoji", "Apple Color Emoji", 'Roboto', Arial, sans-serif;
        }}
        .visualization {{
            margin-bottom: 30px;
            text-align: center;
            background: #fff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #f0f0f0;
            min-height: 350px;
        }}
        .visualization h4 {{
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 15px;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .insights {{
            padding: 20px;
            background-color: #e8f4f8;
            border-left: 5px solid #075e54;
            border-radius: 5px;
        }}
        .insights h4 {{
            color: #075e54;
            margin-bottom: 15px;
        }}
        .insights p {{
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 0;
        }}
        .word-list h3 {{
            font-size: 1.2rem;
            color: #444;
            margin-top: 15px;
        }}
        .word-list ul {{
            list-style: none;
            padding: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .word-list li {{
            background-color: #e9ecef;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
            color: #333;
            border: 1px solid #ced4da;
        }}
        .mini-table-card {{
            background: #fcfdfd;
            border: 1px solid #edf2f1;
            border-radius: 14px;
            padding: 18px;
        }}
        .mini-table-card h4 {{
            color: #075e54;
            margin-bottom: 14px;
            font-size: 1.05rem;
        }}
        .section-note {{
            color: #657271;
            font-size: 0.92rem;
            margin-bottom: 16px;
        }}
        .metric-card {{
            background: linear-gradient(180deg, #ffffff 0%, #f2fbf8 100%);
            border: 1px solid #dcefeb;
            border-radius: 16px;
            padding: 18px;
            text-align: center;
        }}
        .metric-icon {{
            color: #0b7285;
            font-size: 1.3rem;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #063f38;
        }}
        .metric-label {{
            font-size: 0.92rem;
            color: #5b6b6a;
        }}
        .heuristic-badge {{
            font-size: 0.75rem;
            vertical-align: middle;
            margin-left: 8px;
        }}
        details.advanced {{
            margin-top: 16px;
        }}
        details.advanced summary {{
            cursor: pointer;
            font-weight: 700;
            color: #075e54;
            list-style: none;
        }}
        details.advanced summary::-webkit-details-marker {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-meta">
                <div class="profile-img d-flex align-items-center justify-content-center" style="background-color:#0b7a6d;border-radius:50%;width:64px;height:64px;box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                    <span style="font-size:1.6rem;color:#fff;font-weight:700;">{initials}</span>
                </div>
                <div class="header-copy">
                    <h1>WhatsApp Chat Analysis</h1>
                    <p>Detailed report for <strong>{name}</strong></p>
                </div>
            </div>
            <div class="d-flex justify-content-center align-items-center flex-wrap gap-2 mt-2">
                <span class="badge rounded-pill bg-success" style="font-size: 0.9rem; padding: 6px 12px;">
                    <i class="fas {Role Icon} me-1"></i> {Assigned Role}
                </span>
                <span class="status mb-0"><i class="fas fa-circle me-1" style="font-size: 0.8em;"></i>Active Participant</span>
            </div>
            <p class="mb-0 mt-2 px-2" style="opacity: 0.9;">{Role Description}</p>
        </header>
        <div class="row mt-4">
            <div class="col-12">
                <div class="user-report">
                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-gauge-high me-2"></i>Overview</h2>
                        <div class="row g-3">
                            {Overview Cards}
                        </div>
                    </div>

                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-lightbulb me-2"></i>Behavioral Summary <span class="badge bg-secondary heuristic-badge">Heuristic</span></h2>
                        <div class="insights shadow-sm">
                            {Behavioral Insights Text}
                        </div>
                    </div>

                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-calendar-days me-2"></i>Periodic Activity</h2>
                        <div class="row g-4">
                            <div class="col-lg-8">
                                <div class="visualization h-100">
                                    <h4>Monthly Activity Timeline</h4>
                                    {Monthly Activity Timeline}
                                </div>
                            </div>
                            <div class="col-lg-4">
                                <div class="visualization h-100">
                                    <h4>Participation Share Over Time</h4>
                                    {Participation Share Over Time}
                                </div>
                            </div>
                            <div class="col-12">
                                <div class="mini-table-card">
                                    <h4>Periodic Topical Analysis</h4>
                                    <p class="section-note">Month-wise view of this user's activity, keywords, emojis, and content mix.</p>
                                    <div class="table-responsive">
                                        <table class="table table-sm table-striped align-middle mb-0">
                                            <thead>
                                                <tr>
                                                    <th>Month</th>
                                                    <th>Messages</th>
                                                    <th>Busiest Day</th>
                                                    <th>Top Keywords</th>
                                                    <th>Top Emojis</th>
                                                    <th>Media / Links</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Periodic Topical Analysis Rows}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                            <div class="col-12">
                                <div class="mini-table-card">
                                    <h4>Monthly Summary Paragraphs</h4>
                                    <p class="section-note">Generated fully locally from existing sentiment, embedding, keyword, emoji, media, and participation signals.</p>
                                    <div class="row g-4">
                                        {Monthly Summary Cards}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-people-arrows me-2"></i>Interaction</h2>
                        <div class="row g-4">
                            <div class="col-lg-6">
                                <div class="visualization h-100">
                                    <h4>Top Interactions</h4>
                                    {Top Interactions}
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="mini-table-card h-100">
                                    <h4>Interaction Trends Over Time</h4>
                                    <p class="section-note">Recent months and the strongest recurring interaction partners.</p>
                                    <div class="table-responsive">
                                        <table class="table table-sm table-striped align-middle mb-0">
                                            <thead>
                                                <tr>
                                                    <th>Month</th>
                                                    <th>Top Partner</th>
                                                    <th>Interactions</th>
                                                    <th>Top 3 Partners</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Interaction Trends Rows}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-tags me-2"></i>Language And Topics</h2>
                        <div class="row g-4">
                            <div class="col-lg-6">
                                <div class="visualization h-100">
                                    <h4>Content Mix Over Time</h4>
                                    {Content Mix Over Time}
                                </div>
                            </div>
                            <div class="col-lg-6 word-list">
                                <div class="mini-table-card h-100">
                                    <h4>Common Words And Markers</h4>
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <h3><i class="fas fa-cube me-2 text-secondary"></i>Unigrams</h3>
                                            <ul>{Common Unigrams}</ul>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <h3><i class="fas fa-cubes me-2 text-secondary"></i>Bigrams</h3>
                                            <ul>{Common Bigrams}</ul>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <h3><i class="fas fa-layer-group me-2 text-secondary"></i>Trigrams</h3>
                                            <ul>{Common Trigrams}</ul>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <h3><i class="fas fa-exclamation-triangle me-2 text-danger"></i>Hindi Abuse</h3>
                                            <ul>{Hindi Abuse Counts}</ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-balance-scale me-2"></i>Benchmarking (You vs. Group)</h2>
                        <div class="table-responsive">
                            <table class="table table-sm border">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Your Value</th>
                                        <th>Group Average</th>
                                        <th>Performance</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Messages</td>
                                        <td>{Total Messages}</td>
                                        <td>{avg_msgs:.1f}</td>
                                        <td><span class="badge {msg_badge_class}">{msg_performance}</span></td>
                                    </tr>
                                    <tr>
                                        <td>Words/Msg</td>
                                        <td>{Average Message Length:.1f}</td>
                                        <td>{avg_len:.1f}</td>
                                        <td><span class="badge {len_badge_class}">{len_performance}</span></td>
                                    </tr>
                                    <tr>
                                        <td>Response Time</td>
                                        <td>{Average Response Time:.1f}</td>
                                        <td>{avg_speed:.1f}</td>
                                        <td><span class="badge {speed_badge_class}">{speed_performance}</span></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div class="section mb-5">
                        <h2 class="section-title"><i class="fas fa-chart-line me-2"></i>Core Visualizations</h2>
                        <div class="row g-4">      
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Most Active Hours</h4>
                                    {Most Active Hours}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Activity Heatmap</h4>
                                    {Activity Heatmap}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Top Interactions</h4>
                                    {Top Interactions}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Response Time Distribution</h4>
                                    {Response Time Distribution}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Emoji Usage</h4>
                                    {Emoji Usage}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Language Complexity</h4>
                                    {Language Complexity}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="visualization h-100">
                                    <h4>Vocabulary Diversity</h4>
                                    {Vocabulary Diversity}
                                </div>
                            </div>
                        </div>

                        <details class="advanced">
                            <summary>Advanced Analysis</summary>
                            <div class="row g-4 mt-3">
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Big Five Personality Traits <span class="badge bg-secondary heuristic-badge">Heuristic</span></h4>
                                        {Personality Radar Chart}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Personality Evolution <span class="badge bg-secondary heuristic-badge">Heuristic</span></h4>
                                        {Personality Evolution}
                                    </div>
                                </div>
                                <div class="col-md-12">
                                    <div class="visualization h-100">
                                        <h4>Conflict & Tension Heatmap <span class="badge bg-secondary heuristic-badge">Heuristic</span></h4>
                                        {Conflict Heatmap}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Sentiment Over Time</h4>
                                        {Sentiment Over Time}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Sentiment Distribution</h4>
                                        {Sentiment Distribution}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Sentiment (Bubble)</h4>
                                        {Sentiment Bubble}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Language Complexity (POS)</h4>
                                        {Language Complexity POS}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>User Relationship Graph</h4>
                                        {User Relationship Graph}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Skills Radar Chart <span class="badge bg-secondary heuristic-badge">Heuristic</span></h4>
                                        {Skills Radar Chart}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Emotion Trends (Time Series)</h4>
                                        {Emotion Over Time}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="visualization h-100">
                                        <h4>Word Cloud</h4>
                                        {Word Cloud}
                                    </div>
                                </div>
                            </div>
                        </details>
                    </div>

                    <div class="section">
                        <h2 class="section-title"><i class="fas fa-table me-2"></i>Detailed Stats</h2>
                        <details class="advanced" open>
                            <summary>Expand Full Metrics</summary>
                            <div class="table-responsive mt-3">
                                <table class="table table-striped table-hover border">
                                    <tbody>
                                        <tr><th>Total Messages</th><td>{Total Messages}</td></tr>
                                        <tr><th>Total Words</th><td>{Total Words}</td></tr>
                                        <tr><th>Unique Words</th><td>{Unique Words Count}</td></tr>
                                        <tr><th>Total Emojis</th><td>{Total Emojis}</td></tr>
                                        <tr><th>Top 5 Emojis</th><td class="emoji">{top_5_emojis_html}</td></tr>
                                        <tr><th>Total URLs</th><td>{Total URLs}</td></tr>
                                        <tr><th>Total YouTube URLs</th><td>{Total YouTube URLs}</td></tr>
                                        <tr><th>Total Media</th><td>{Total Media}</td></tr>
                                        <tr><th>Total Edits</th><td>{Total Edits}</td></tr>
                                        <tr><th>Total Deletes</th><td>{Total Deletes}</td></tr>
                                        <tr><th>Average Message Length</th><td>{Average Message Length:.2f}</td></tr>
                                        <tr><th>Average Sentence Length</th><td>{Average Sentence Length:.2f}</td></tr>
                                        <tr><th>Positive Messages</th><td><span class="badge bg-success">{Positive Messages}</span></td></tr>
                                        <tr><th>Negative Messages</th><td><span class="badge bg-danger">{Negative Messages}</span></td></tr>
                                        <tr><th>Morning Messages</th><td>{Morning Messages}</td></tr>
                                        <tr><th>Mid-day Messages</th><td>{Mid-day Messages}</td></tr>
                                        <tr><th>Evening Messages</th><td>{Evening Messages}</td></tr>
                                        <tr><th>Night Messages</th><td>{Night Messages}</td></tr>
                                        <tr><th>Most Active Period</th><td><strong>{Most Active Period}</strong></td></tr>
                                        <tr><th>Average Response Time (mins)</th><td>{Average Response Time:.2f}</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </details>
                    </div>
                </div>
            </div>
        </div>
        <footer class="footer">
            <p class="mb-1">Generated with <i class="fas fa-heart text-danger"></i> by WhatsApp Analyzer</p>
            <p class="mb-0"><a href="https://github.com/gauravmeena0708/whatsapp-analyzer" target="_blank"><i class="fab fa-github me-1"></i>Visit the Project</a></p>
        </footer>
    </div>
</body>
</html>
"""

index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhatsApp Chat Analysis - Group Summary</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Roboto', sans-serif; background: linear-gradient(180deg, #eef7f5 0%, #f8f9fa 100%); color: #333; }}
        .container {{ max-width: 1320px; margin: 30px auto; }}
        .header {{ background: linear-gradient(135deg, #075e54, #0b7285); color: white; padding: 34px 24px; border-radius: 18px; text-align: center; box-shadow: 0 14px 30px rgba(7,94,84,0.18); }}
        .header p {{ opacity: 0.9; margin-top: 10px; }}
        .section {{ background: white; padding: 28px; margin-top: 20px; border-radius: 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.06); }}
        .section h2 {{ color: #075e54; border-bottom: 2px solid #ececec; padding-bottom: 10px; margin-bottom: 20px; }}
        .metric-card {{ background: linear-gradient(180deg, #ffffff 0%, #f2fbf8 100%); border: 1px solid #dcefeb; border-radius: 16px; padding: 18px; text-align: center; }}
        .metric-icon {{ color: #0b7285; font-size: 1.4rem; margin-bottom: 8px; }}
        .metric-value {{ font-size: 1.6rem; font-weight: 700; color: #063f38; }}
        .metric-label {{ font-size: 0.95rem; color: #5b6b6a; }}
        .highlight-card {{ border-radius: 16px; padding: 18px; background: #f7fbfa; border: 1px solid #e4efec; }}
        .highlight-kicker {{ text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.75rem; color: #0b7285; font-weight: 700; }}
        .highlight-name {{ font-size: 1.25rem; font-weight: 700; margin-top: 6px; color: #063f38; }}
        .highlight-detail {{ color: #61706f; margin-top: 6px; }}
        .user-link {{ text-decoration: none; color: #075e54; font-weight: bold; display: flex; align-items: center; justify-content: center; height: 100%; }}
        .user-card {{ border: 1px solid #e0e0e0; padding: 16px; margin-bottom: 15px; border-radius: 12px; transition: all 0.25s ease; background-color: #fafafa; height: 100%; min-height: 88px; text-align: center; }}
        .user-card:hover {{ background-color: #e8f4f8; border-color: #075e54; transform: translateY(-3px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }}
        .participant-name {{ display: block; font-size: 1rem; }}
        .participant-meta {{ display: block; font-size: 0.82rem; margin-top: 4px; color: #5b6b6a; font-weight: 500; }}
        .visualization {{ text-align: center; padding: 16px; border: 1px solid #f0f0f0; border-radius: 14px; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); min-height: 350px; }}
        .visualization h4 {{ color: #555; font-size: 1.1rem; margin-bottom: 15px; }}
        .visualization img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        .mini-table-card {{ background: #fcfdfd; border: 1px solid #edf2f1; border-radius: 14px; padding: 18px; }}
        .mini-table-card h4 {{ color: #075e54; margin-bottom: 14px; font-size: 1.05rem; }}
        .topic-card {{ background: #fbfcfc; border: 1px solid #edf2f1; border-radius: 14px; padding: 18px; height: 100%; }}
        .topic-card h4 {{ color: #075e54; font-size: 1.05rem; margin-bottom: 12px; }}
        .topic-card ul {{ list-style: none; padding-left: 0; margin-bottom: 0; }}
        .topic-card li {{ padding: 6px 10px; margin-bottom: 8px; background: #eef7f5; border-radius: 999px; display: inline-block; margin-right: 8px; }}
        .topic-chip {{ padding: 8px 12px; background: #eef7f5; border-radius: 999px; display: inline-block; margin: 0 8px 8px 0; }}
        .insights {{ padding: 20px; background-color: #e8f4f8; border-left: 5px solid #075e54; border-radius: 5px; }}
        .footer {{ background-color: #075e54; color: white; text-align: center; padding: 20px; margin-top: 30px; border-radius: 18px; box-shadow: 0 -4px 8px rgba(0,0,0,0.08); }}
        .table th {{ background-color: #075e54; color: #fff; font-weight: 500; }}
        .section-note {{ color: #657271; font-size: 0.92rem; margin-bottom: 16px; }}
        .heuristic-badge {{ font-size: 0.75rem; vertical-align: middle; margin-left: 8px; }}
        details.advanced {{ margin-top: 16px; }}
        details.advanced summary {{ cursor: pointer; font-weight: 700; color: #075e54; list-style: none; }}
        details.advanced summary::-webkit-details-marker {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1 class="mb-0"><i class="fab fa-whatsapp me-3"></i>WhatsApp Group Chat Summary</h1>
            <p class="mb-0">High-level group patterns, participant roles, activity timeline, interactions, and birthday tracking.</p>
        </header>

        <div class="section">
            <h2><i class="fas fa-gauge-high me-2"></i>Overview</h2>
            <div class="row g-3">
                {overview_cards}
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-file-lines me-2"></i>Group Narrative</h2>
            <div class="insights shadow-sm">
                {group_narrative_summary}
            </div>
        </div>
        
        <div class="section">
            <h2><i class="fas fa-users me-2"></i>Group Statistics</h2>
            <div class="table-responsive">
                <table class="table table-striped table-hover border">
                    <tbody>
                        {expanded_group_stats_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-calendar-days me-2"></i>Period-Wise Summary</h2>
            <div class="row g-4">
                <div class="col-lg-8">
                    <div class="visualization h-100">
                        <h4>Monthly Activity Timeline</h4>
                        {group_activity_timeline}
                    </div>
                </div>
                <div class="col-lg-4">
                    <div class="mini-table-card h-100">
                        <h4>Period Snapshot</h4>
                        <div class="table-responsive">
                            <table class="table table-sm table-striped align-middle mb-0">
                                <tbody>
                                    {period_summary_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row g-4 mt-1">
                <div class="col-12">
                    <div class="mini-table-card">
                        <h4>Periodic Topical Analysis</h4>
                        <p class="section-note">Recent active months with the dominant participant, busiest day, recurring keywords, and top emojis for that month.</p>
                        <div class="table-responsive">
                            <table class="table table-sm table-striped align-middle mb-0">
                                <thead>
                                    <tr>
                                        <th>Month</th>
                                        <th>Messages</th>
                                        <th>Top User</th>
                                        <th>Busiest Day</th>
                                        <th>Top Keywords</th>
                                        <th>Top Emojis</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {periodic_topic_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row g-4 mt-1">
                <div class="col-12">
                    <div class="mini-table-card">
                        <h4>Monthly Summary Paragraphs</h4>
                        <p class="section-note">Generated fully locally from existing sentiment, embedding, keyword, emoji, media, and participation signals.</p>
                        <div class="row g-4">
                            {monthly_summary_cards}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-star me-2"></i>Group Highlights</h2>
            <div class="row g-3">
                {group_highlights}
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-ranking-star me-2"></i>Leaderboards</h2>
            <div class="row g-4">
                {leaderboard_tables}
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-masks-theater me-2"></i>Participant Roles <span class="badge bg-secondary heuristic-badge">Heuristic</span></h2>
            <p class="section-note">Role labels are heuristic and based on activity patterns, messaging style, and social behavior in the group.</p>
            <div class="table-responsive">
                <table class="table table-striped table-hover border align-middle">
                    <thead>
                        <tr>
                            <th>User</th>
                            <th>Role</th>
                            <th>Most Active Period</th>
                            <th>Messages</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        {role_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-people-arrows me-2"></i>Interaction Summary</h2>
            <div class="row g-4">
                {interaction_summary}
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-tags me-2"></i>Topics And Expression</h2>
            <div class="row g-4">
                <div class="col-lg-3">
                    <div class="topic-card">
                        <h4>Top Emojis</h4>
                        <div>{top_group_emojis}</div>
                    </div>
                </div>
                <div class="col-lg-3">
                    <div class="topic-card">
                        <h4>Common Unigrams</h4>
                        <ul>{common_unigrams}</ul>
                    </div>
                </div>
                <div class="col-lg-3">
                    <div class="topic-card">
                        <h4>Common Bigrams</h4>
                        <ul>{common_bigrams}</ul>
                    </div>
                </div>
                <div class="col-lg-3">
                    <div class="topic-card">
                        <h4>Common Trigrams</h4>
                        <ul>{common_trigrams}</ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-cake-candles me-2"></i>Birthdays <span class="badge bg-secondary heuristic-badge">Manual + Inferred</span></h2>
            <div class="row g-4">
                <div class="col-lg-5">
                    <div class="mini-table-card h-100">
                        <h4>Manual Birthday Registry</h4>
                        <p class="section-note">{manual_birthday_note}</p>
                        <div class="table-responsive">
                            <table class="table table-sm table-striped align-middle mb-0">
                                <thead>
                                    <tr><th>User</th><th>Date</th><th>Source</th></tr>
                                </thead>
                                <tbody>
                                    {manual_birthday_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                <div class="col-lg-7">
                    <div class="mini-table-card h-100">
                        <h4>Probable Birthdays Inferred From Chat</h4>
                        <p class="section-note">{inferred_birthday_note}</p>
                        <div class="table-responsive">
                            <table class="table table-sm table-striped align-middle mb-0">
                                <thead>
                                    <tr><th>User</th><th>Likely Date</th><th>Confidence</th><th>Evidence Count</th><th>Example Evidence</th></tr>
                                </thead>
                                <tbody>
                                    {inferred_birthday_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-address-book me-2"></i>Participants</h2>
            <p class="text-muted mb-4">Click on a participant to view their detailed individual analysis report.</p>
            <div class="row g-3">
                {user_links}
            </div>
        </div>

        <div class="section">
            <h2><i class="fas fa-chart-area me-2"></i>Advanced Visualizations</h2>
            <details class="advanced">
                <summary>Expand Group Visualizations</summary>
                <div class="row g-4 mt-3">
                    <div class="col-md-6">
                        <div class="visualization h-100">
                            <h4>Activity Heatmap</h4>
                            {Activity Heatmap}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="visualization h-100">
                            <h4>User Relationship Graph</h4>
                            {User Relationship Graph}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="visualization h-100">
                            <h4>Most Active Hours</h4>
                            {Most Active Hours}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="visualization h-100">
                            <h4>Sentiment Over Time</h4>
                            {Sentiment Over Time}
                        </div>
                    </div>
                </div>
            </details>
        </div>

        <footer class="footer">
            <p class="mb-0">Generated with <i class="fas fa-heart text-danger"></i> by WhatsApp Analyzer</p>
        </footer>
    </div>
</body>
</html>
"""
