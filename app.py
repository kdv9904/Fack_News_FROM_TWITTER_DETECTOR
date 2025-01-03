from flask import Flask, request, render_template, jsonify
import joblib
import requests
import pandas as pd

app = Flask(__name__)

model = joblib.load('fake_news_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# API key for NewsAPI
api_key = '41400d6dcdaa46fd8fb81df8714cd0b7'
base_url = 'https://newsapi.org/v2/top-headlines?country=in&apiKey=' + api_key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'news' in request.form:
        news = request.form['news']
        text_vectorized = vectorizer.transform([news])
        prediction = model.predict(text_vectorized)
        result = 'Fake' if prediction[0] == 1 else 'True'
    else:
        result = 'No news provided'
    
    # Fetch articles from NewsAPI
    response = requests.get(base_url)
    articles = response.json().get('articles', [])
    
    return render_template('result.html', prediction=result, articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
