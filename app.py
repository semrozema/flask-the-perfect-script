import os
import joblib
import string
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load models and resources
ridge_revenue_model = joblib.load('./models/ridge_revenue_model.pkl')
ridge_rating_model = joblib.load('./models/ridge_rating_model.pkl')
doc2vec_model = Doc2Vec.load('./models/doc2vec_model.d2v')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesses the text: lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

def vectorize_text(text_tokens):
    """Infers a vector for the text using the trained Doc2Vec model."""
    return doc2vec_model.infer_vector(text_tokens)

def predict_revenue_and_rating(text):
    """Preprocesses and vectorizes text, then predicts revenue and rating."""
    processed_text = preprocess_text(text)
    text_vector = vectorize_text(processed_text).reshape(1, -1)
    predicted_revenue = ridge_revenue_model.predict(text_vector)[0]
    predicted_rating = ridge_rating_model.predict(text_vector)[0]
    return predicted_revenue, predicted_rating

# Define routes
@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        text = file.read().decode('utf-8')
        predicted_revenue, predicted_rating = predict_revenue_and_rating(text)
        return render_template('result.html', revenue=predicted_revenue, rating=predicted_rating)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
