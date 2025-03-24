from flask import Flask, request, jsonify, send_from_directory
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from gtts import gTTS
import pandas as pd
import os
import uuid
from keybert import KeyBERT
from translate import Translator
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

kw_model = KeyBERT()

@app.route('/')
def home():
    return 'Welcome to the News summarization API'

# News Extraction Function
def fetch_news_articles(company_name):
    urls = [
        f"https://www.bbc.com/search?q={company_name}&page=0",
        f"https://www.bbc.com/search?q={company_name}&page=1",
        f"https://www.bbc.com/search?q={company_name}&page=2",
        f"https://www.bbc.com/search?q={company_name}&page=3",
        f"https://www.bbc.com/search?q={company_name}&page=4",
        f"https://www.bbc.com/search?q={company_name}&page=5"
    ]
    articles = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for item in soup.find_all('div', class_='sc-c6f6255e-0 eGcloy'):
                title_element = item.find('h2')
                summary_element = item.find('div', class_='sc-4ea10043-3 kMizuB')
                if title_element and summary_element:
                    title = title_element.text
                    summary = summary_element.text
                    articles.append({
                        "Title": title,
                        "Summary": summary,
                        "Sentiment": "",
                        "Topics": []  # Placeholder for topic extraction
                    })
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue
    
    return articles[:15]

# Sentiment Analysis Function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"
    
# Comparative Analysis Function
def comparative_analysis(articles):
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        sentiment_counts[article["Sentiment"]] += 1
    return sentiment_counts

# Generate Comparative Insights
def generate_comparative_insights(articles):
    if len(articles) < 2:
        return []
    insights = []
    for i in range(len(articles) - 1):
        comparison = f"Article {i + 1} highlights {articles[i]['Title']}, while Article {i + 2} discusses {articles[i + 1]['Title']}."
        impact = f"The first article focuses on {articles[i]['Sentiment']} aspects, while the second highlights {articles[i + 1]['Sentiment']} aspects."
        insights.append({"Comparison": comparison, "Impact": impact})
    return insights   

# Extracting important keywords
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    return [keyword for keyword, score in keywords]

# Function to translate text to Hindi and generate speech
def generate_hindi_audio(text):
    try:
        translator = Translator(to_lang="hi")
        translated_text = translator.translate(text)
        
        if not os.path.exists('static'):
            os.makedirs('static')
        
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join('static', audio_filename)
        tts = gTTS(text=translated_text, lang="hi")
        tts.save(audio_path)
        
        return f"http://127.0.0.1:5000/tts/{audio_filename}"
    except Exception as e:
        print(f"Error in translation or audio generation: {e}")
        return None

@app.route('/analyze-news', methods=['POST'])
def analyze_news():
    try:
        data = request.json
        company_name = data.get('company_name')
        
        if not company_name:
            return jsonify({"error": "Company name is required"}), 400
        
        articles = fetch_news_articles(company_name)
        
        if not articles:
            return jsonify({"error": "No articles found"}), 404
        
        for article in articles:
            article["Sentiment"] = analyze_sentiment(article["Summary"])
            article["key_words"] = extract_keywords(article["Summary"])
            article["Hindi_Audio"] = generate_hindi_audio(article["Summary"])
        
        sentiment_counts = comparative_analysis(articles)
        comparative_insights = generate_comparative_insights(articles)
        
        response = {
            "Company": company_name,
            "Articles": articles,
            "Comparative Sentiment Score": {
                "Sentiment Distribution": sentiment_counts,
                "Coverage Differences": comparative_insights,
                "Topic Overlap": {
                    "Common Topics": list(set(articles[0]["Topics"]).intersection(articles[1]["Topics"])),
                    "Unique Topics in Article 1": list(set(articles[0]["Topics"]).difference(articles[1]["Topics"])),
                    "Unique Topics in Article 2": list(set(articles[1]["Topics"]).difference(articles[0]["Topics"]))
                }
            },
            "Final Sentiment Analysis": f"{company_name}'s latest news coverage is mostly {max(sentiment_counts, key=sentiment_counts.get)}.",
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tts/<filename>')
def serve_tts(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)