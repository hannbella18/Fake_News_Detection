import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train_tfidf, X_val_tfidf, y_train, y_val, vectorizer = load_and_preprocess_data('train.csv', 'test.csv')

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)

# Make predictions with Logistic Regression
y_pred_log_reg = log_reg.predict(X_val_tfidf)

# Evaluate the Logistic Regression model
accuracy_log = accuracy_score(y_val, y_pred_log_reg)
precision_log = precision_score(y_val, y_pred_log_reg)
recall_log = recall_score(y_val, y_pred_log_reg)
f1_log = f1_score(y_val, y_pred_log_reg)

print(f"Logistic Regression - Accuracy: {accuracy_log:.4f}, Precision: {precision_log:.4f}, Recall: {recall_log:.4f}, F1 Score: {f1_log:.4f}")

# Save Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)
print("Logistic Regression model saved.")

# Train SVM model
svm = SVC()
svm.fit(X_train_tfidf, y_train)

# Make predictions with SVM
y_pred_svm = svm.predict(X_val_tfidf)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_val, y_pred_svm)
precision_svm = precision_score(y_val, y_pred_svm)
recall_svm = recall_score(y_val, y_pred_svm)
f1_svm = f1_score(y_val, y_pred_svm)

# Save SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
print("SVM model saved.")

# Save TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("TF-IDF vectorizer saved.")

print(f"SVM - Accuracy: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, F1 Score: {f1_svm:.4f}")

