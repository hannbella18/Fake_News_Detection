from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load data and models
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Preprocess data
data_fake["class"] = 0
data_true["class"] = 1

# Use a smaller subset of data for faster processing
data_fake = data_fake.sample(frac=0.1, random_state=1)
data_true = data_true.sample(frac=0.1, random_state=1)

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)


def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


data['text'] = data['text'].apply(wordopt)
x = data['text']
y = data['class']

vectorization = TfidfVectorizer()
xv = vectorization.fit_transform(x)

# Train models
LR = LogisticRegression(max_iter=100)
LR.fit(xv, y)
DT = DecisionTreeClassifier(max_depth=10)  # Reduced depth for quicker training
DT.fit(xv, y)
GB = GradientBoostingClassifier(random_state=0, n_estimators=10)  # Reduced n_estimators for quicker training
GB.fit(xv, y)
RF = RandomForestClassifier(random_state=0, n_estimators=10)  # Reduced n_estimators for quicker training
RF.fit(xv, y)


def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"


def fetch_text_from_link(link):
    try:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        return str(e)


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data.get('text', '')
    news_link = data.get('link', '')

    if news_link:
        news_text = fetch_text_from_link(news_link)

    if not news_text:
        return jsonify({'error': 'No text provided'}), 400

    processed_text = wordopt(news_text)
    new_xv_test = vectorization.transform([processed_text])

    pred_LR = output_label(LR.predict(new_xv_test)[0])
    pred_DT = output_label(DT.predict(new_xv_test)[0])
    pred_GB = output_label(GB.predict(new_xv_test)[0])
    pred_RF = output_label(RF.predict(new_xv_test)[0])

    return jsonify({
        'pred_LR': pred_LR,
        'pred_DT': pred_DT,
        'pred_GB': pred_GB,
        'pred_RF': pred_RF
    })


def evaluate_model(model, xv_test, y_test):
    predictions = model.predict(xv_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return {
        'model': model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Use part of the original dataset as the test set
    test_data_fake = pd.read_csv('Fake.csv').sample(frac=0.1, random_state=2)
    test_data_true = pd.read_csv('True.csv').sample(frac=0.1, random_state=2)

    test_data_fake["class"] = 0
    test_data_true["class"] = 1

    test_data = pd.concat([test_data_fake, test_data_true], axis=0)
    test_data = test_data.drop(['title', 'subject', 'date'], axis=1)
    test_data['text'] = test_data['text'].apply(wordopt)

    x_test = test_data['text']
    y_test = test_data['class']

    xv_test = vectorization.transform(x_test)

    lr_results = evaluate_model(LR, xv_test, y_test)
    dt_results = evaluate_model(DT, xv_test, y_test)
    gb_results = evaluate_model(GB, xv_test, y_test)
    rf_results = evaluate_model(RF, xv_test, y_test)

    results = {
        'Logistic Regression': lr_results,
        'Decision Tree': dt_results,
        'Gradient Boosting': gb_results,
        'Random Forest': rf_results
    }

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
