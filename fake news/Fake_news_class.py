from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

# Initialize the Sai class
from fak_trails import sai
fake_news_detector = sai()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_fake_news():
    if request.method == 'POST':
        text = request.form['text']

        # Use the fake_news24 method to detect fake news
        result = fake_news_detector.fake_news24(text)

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
