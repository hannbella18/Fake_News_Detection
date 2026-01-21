import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data

print("Starting data loading and preprocessing...")

# Load and preprocess data
X_train_tfidf, X_val_tfidf, y_train, y_val, vectorizer = load_and_preprocess_data('train.csv', 'test.csv')

print("Data loading and preprocessing completed.")
print(f"Training data shape: {X_train_tfidf.shape}")
print(f"Validation data shape: {X_val_tfidf.shape}")

# Encode labels to integers
encoder = LabelEncoder()
y_train_nn = encoder.fit_transform(y_train)
y_val_nn = encoder.transform(y_val)

print("Starting Logistic Regression training...")

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

print("Logistic Regression training completed.")

print("Starting SVM training...")

# Train SVM model with linear kernel for faster training
svm = SVC(kernel='linear', probability=True, verbose=True)
# Use a smaller subset of the data for faster training
subset_size = 5000
X_train_subset = X_train_tfidf[:subset_size]
y_train_subset = y_train[:subset_size]
print(f"Training SVM with subset size: {subset_size}")
svm.fit(X_train_subset, y_train_subset)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

print("SVM training completed.")

print("Starting Neural Network training...")

# Define and train Neural Network model
model = Sequential()
model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_tfidf, y_train_nn, epochs=5, batch_size=128, validation_data=(X_val_tfidf, y_val_nn))
model.save('nn_model.h5')

print("Neural Network training completed.")

print("Starting model evaluations...")

# Evaluate models
y_pred_log_reg = log_reg.predict(X_val_tfidf)
y_pred_svm = svm.predict(X_val_tfidf)
y_pred_nn = (model.predict(X_val_tfidf) > 0.5).astype("int32")

# Logistic Regression evaluation
accuracy_log = accuracy_score(y_val, y_pred_log_reg)
precision_log = precision_score(y_val, y_pred_log_reg)
recall_log = recall_score(y_val, y_pred_log_reg)
f1_log = f1_score(y_val, y_pred_log_reg)
print(f"Logistic Regression - Accuracy: {accuracy_log:.4f}, Precision: {precision_log:.4f}, Recall: {recall_log:.4f}, F1 Score: {f1_log:.4f}")

# SVM evaluation
accuracy_svm = accuracy_score(y_val, y_pred_svm)
precision_svm = precision_score(y_val, y_pred_svm)
recall_svm = recall_score(y_val, y_pred_svm)
f1_svm = f1_score(y_val, y_pred_svm)
print(f"SVM - Accuracy: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, F1 Score: {f1_svm:.4f}")

# Neural Network evaluation
accuracy_nn = accuracy_score(y_val_nn, y_pred_nn)
precision_nn = precision_score(y_val_nn, y_pred_nn)
recall_nn = recall_score(y_val_nn, y_pred_nn)
f1_nn = f1_score(y_val_nn, y_pred_nn)
print(f"Neural Network - Accuracy: {accuracy_nn:.4f}, Precision: {precision_nn:.4f}, Recall: {recall_nn:.4f}, F1 Score: {f1_nn:.4f}")

print("Model evaluations completed.")

# Save vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("All models and vectorizer saved.")
