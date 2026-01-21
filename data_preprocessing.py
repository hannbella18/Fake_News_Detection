import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import re

def preprocess_text_simple(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

def load_and_preprocess_data(train_path, test_path):
    # Load the datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Handle missing values
    imputer = SimpleImputer(strategy='constant', fill_value='')
    train_df['title'] = imputer.fit_transform(train_df[['title']]).ravel()
    train_df['author'] = imputer.fit_transform(train_df[['author']]).ravel()
    train_df['text'] = imputer.fit_transform(train_df[['text']]).ravel()

    # Combine title, author, and text columns into a single column for processing
    train_df['combined'] = train_df['title'] + ' ' + train_df['author'] + ' ' + train_df['text']
    train_df['combined'] = train_df['combined'].apply(preprocess_text_simple)

    # Split the data into features and labels
    X = train_df['combined']
    y = train_df['label']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    return X_train_tfidf, X_val_tfidf, y_train, y_val, vectorizer

if __name__ == "__main__":
    X_train_tfidf, X_val_tfidf, y_train, y_val, vectorizer = load_and_preprocess_data('train.csv', 'test.csv')
    print("Data preprocessing completed.")
