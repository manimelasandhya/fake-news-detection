# Import necessary libraries
import joblib
from cffi import model
from flask import Flask, render_template, request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a Flask app
app = Flask(__name__)

# Load the pre-trained model
loaded_model = joblib.load('fakenews.pkl')


# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        news_text = request.form['news_text']


        tfidf_test = tfidf_vectorizer.transform(news_text)
        #print(tfidf_test)
        processed_text_2d = np.array(tfidf_test).reshape(1, -1)
        # Preprocess the news_text (you can implement your own preprocessing here)
        # Example: You might need to tokenize, clean, and vectorize the text.
        # Then, pass the processed text to the model for prediction.
        prediction = loaded_model.predict(processed_text_2d)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
