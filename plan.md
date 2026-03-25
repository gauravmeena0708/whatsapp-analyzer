# WhatsApp Chat Analyzer: Advanced Psychological Profiling Plan

This plan outlines the evolution of the WhatsApp Chat Analyzer from a statistical tool to a deep psychological and personality profiling engine.

## 1. Personality Mapping (The OCEAN Model)
Implement the **Big Five Personality Traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) using linguistic markers.

*   **Extraversion**: High message frequency, frequent use of first-person plural pronouns ("we", "us"), high emoji/exclamation density.
*   **Agreeableness**: High frequency of positive emotion words, low "abuse" count, use of inclusive language.
*   **Neuroticism**: Higher frequency of negative emotion words, frequent "first-person singular" pronouns ("I", "me", "mine").
*   **Conscientiousness**: Longer average message length, objective/formal communication style, structured language.
*   **Openness**: Use of diverse vocabulary (Lexical Diversity), mentions of abstract concepts or new ideas.

## 2. Social Dynamics & Relationship Mapping
Go beyond simple "mentions" to understand the hierarchy and bonds within the group.

*   **Reciprocity Score**: A matrix calculating the balance of replies between User A and User B.
*   **Initiation Rate**: Identifying who starts conversations vs. who only joins existing ones.
*   **Response Latency Matrix**: Measuring bond strength based on how quickly User A responds to User B vs. other participants.
*   **Influence Score**: Calculating how much a user's message triggers a "chain reaction" of replies.

## 3. Automated Group Role Identification
Assign roles based on data-driven psychological signatures:
*   **The Mediator**: High response rate, high agreeableness, inclusive language.
*   **The Energizer**: High emoji density, most active during peak hours, high exclamation count.
*   **The Lurker**: High participation in active periods with low word count/frequency.
*   **The Expert**: High technical keyword matches, long/structured sentences, low response latency (thoughtful).

## 4. Advanced Sentiment & Contextual Intelligence
*   **VAD Model**: Implement Valence-Arousal-Dominance to distinguish between "Excited" (High Arousal) and "Calm" (Low Arousal) positivity.
*   **Sarcasm & Banter Detection**: Correlate playful emojis (😜, 😂) with negative sentiment markers to identify friendly teasing vs. genuine conflict.
*   **Conflict Heatmap**: Identifying periods of high tension based on sentiment drops and rapid-fire short messages.

## 5. Technical & Interactive Enhancements
*   **Interactive Visualizations**: Replace static Matplotlib PNGs with **Plotly** or **Chart.js** for hover-over data and zooming.
*   **Comparative Benchmarking**: Add "User vs. Group Average" metrics to every report to provide context.
*   **Temporal Evolution**: A "Personality Over Time" graph showing how a user's communication style shifts during different months or life events.

## 6. Implementation Strategy
1.  **Update `analysis_utils.py`**: Add the personality scoring and relationship matrix logic.
2.  **Update `constants.py`**: Add Big Five weightings and psychological keyword lists.
3.  **Refactor `analyzer.py`**: Update the reporting logic to handle group-wide comparative data.
4.  **Modernize `html_template`**: Integrate JS-based charting libraries for a more professional dashboard feel.
