from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train_tfidf, X_val_tfidf, y_train, y_val, vectorizer = load_and_preprocess_data('train.csv', 'test.csv')

# Encode labels to integers
encoder = LabelEncoder()
y_train_nn = encoder.fit_transform(y_train)
y_val_nn = encoder.transform(y_val)

# Define the neural network model
model = Sequential()
model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf, y_train_nn, epochs=5, batch_size=128, validation_data=(X_val_tfidf, y_val_nn))

# Make predictions
y_pred_nn = (model.predict(X_val_tfidf) > 0.5).astype("int32")

# Evaluate the model
accuracy = accuracy_score(y_val_nn, y_pred_nn)
precision = precision_score(y_val_nn, y_pred_nn)
recall = recall_score(y_val_nn, y_pred_nn)
f1 = f1_score(y_val_nn, y_pred_nn)

print(f"Neural Network - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Save Neural Network model
model.save('nn_model.h5')

print("Neural network model saved successfully.")
