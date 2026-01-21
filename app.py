from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import re

app = Flask(__name__)

# Define the paths to the model files
logistic_model_path = 'logistic_regression_model.pkl'
svm_model_path = 'svm_model.pkl'
nn_model_path = 'nn_model.h5'
vectorizer_path = 'tfidf_vectorizer.pkl'

# Load the trained models
with open(logistic_model_path, 'rb') as f:
    logistic_model = pickle.load(f)
with open(svm_model_path, 'rb') as f:
    svm_model = pickle.load(f)

# Load the neural network model
nn_model = load_model(nn_model_path)

# Load the TF-IDF vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocess text function
def preprocess_text_simple(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_article = request.form['news_article']
    preprocessed_article = preprocess_text_simple(news_article)
    vectorized_article = vectorizer.transform([preprocessed_article])

    # Get predictions from all models
    logistic_pred = logistic_model.predict(vectorized_article)[0]
    svm_pred = svm_model.predict(vectorized_article)[0]
    nn_pred = (nn_model.predict(vectorized_article.toarray()) > 0.5).astype("int32")[0][0]

    # Combine predictions and get confidence score
    predictions = np.array([logistic_pred, svm_pred, nn_pred])
    final_prediction = np.bincount(predictions).argmax()
    confidence = np.mean(predictions == final_prediction)

    prediction_label = 'Fake' if final_prediction == 1 else 'Real'

    return render_template('index.html', prediction=prediction_label, confidence=f'{confidence * 100:.2f}%')

if __name__ == '__main__':
    app.run(debug=True)
