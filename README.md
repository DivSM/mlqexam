# Sentiment Analysis Webapp

<div>
  <p>
    This is a simple single-page Flask webapp used to showcase the predicted sentiment by the trained machine learning model for any text entered.
    <br />
    Important files and folders:
    <br />
    (1) app.py = Main code to run the single-page webapp
    <br />
    (2) logr_m.pickle = Trained model used to predict sentiment
    <br />
    (3) tfidf_vectorizer = Trained vectorizer used to predict sentiment
    <br />
    (4) requirements.txt = Lists the packages that need to be installed
    <br />
    (5) home.html under templates = HTML & CSS code for the webapp
    <br />
    (6) img_pool folder under static = images of emojis for sentiments
    <br />
    (7) styling.css under static = CSS styling code for home.html
    <br />
    (8) research folder = Twitter data used for training as obtained from kaggle and the colab notebook used for training model
    <br />
    <a href="https://www.kaggle.com/datasets/milobele/sentiment140-dataset-1600000-tweets/data">Twitter Data</a>
  </p>
  <br />
  <br />
  <br />
  <p>
    The app classifies the text as Positive, Neutral or Negative. 
    <br />
    To run the app after downloading the files:
    <br />
    (1) Install the required packages: pip install -r requirements.txt
    <br />
    (2) Start the app: python app.py
    <br />
    The app shall run locally on http://127.0.0.1:5000  
  </p>
</div>