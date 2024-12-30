from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model, tfidf
    # Load the model
    with open('logr_m.pickle', 'rb') as f:
        model = pickle.load(f)
    # Load the TF-IDF vectorizer
    with open('tfidf_vectorizer.pickle', 'rb') as f:
        tfidf = pickle.load(f)

######################### Code for Sentiment Analysis ##########################
@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = ''
    img_filename = ''
    text = ''

    if request.method == 'POST':
        text = request.form['text']

        # Processing text
        def clean_text(text):
            pat1 = r'@[^ ]+'  # @ signs and value
            pat2 = r'https?://[A-Za-z0-9./]+'  # links
            pat3 = r'\'s'  # floating s's
            pat4 = r'\#\w+'  # hashtags and value
            pat5 = r'&amp '  # & and
            pat6 = r'[^A-Za-z\s]'  # remove non-alphabet
            combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5, pat6))
            text = re.sub(combined_pat, "", text).lower()
            return text.strip()

        text = clean_text(text)

        lem = WordNetLemmatizer()

        def tokenize_lem(sentence):
            outlist = []
            token = sentence.split()
            for tok in token:
                outlist.append(lem.lemmatize(tok))
            return " ".join(outlist)

        text = tokenize_lem(text)

        # Create TF-IDF vector
        tweet_tfidf = tfidf.transform([text])

        # Manual scaling based on data obtained from colab notebook
        min_len = 6
        max_len = 375
        single_tweet_len_scaled = (len(text) - min_len) / (max_len - min_len)
        single_tweet_len_scaled = np.array([[single_tweet_len_scaled]])

        # Combine features
        tweet_features = scipy.sparse.hstack([tweet_tfidf, single_tweet_len_scaled], format="csr")

        # Predicting with model
        prediction = model.predict(tweet_features)
        print(prediction)  # Output: [-1] or [0] or [1]

        # Use match-case to set sentiment and image based on the prediction
        match prediction:
            case -1:
                sentiment = 'Negative'
                img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
            case 0:
                sentiment = 'Neutral'
                img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral.png')
            case 1:
                sentiment = 'Positive'
                img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'smiling.png')
            case _:
                sentiment = 'Error: None'
                img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral.png')

    # Rendering the home2.html template with the results
    return render_template('home.html', text=text, sentiment=sentiment, image=img_filename)
######################### Code for Sentiment Analysis ##########################

if __name__ == "__main__":
    init()
    app.run()