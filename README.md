# 📰 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/AI-Scikit_Learn-orange)

**CSC3600 Project** is a Machine Learning–based web application developed to classify news articles as **Fake News** or **Not a Fake News (Real)**.

---

## 🚀 Key Features

- 🔍 **Multi-Model Prediction**  
  Uses four machine learning algorithms:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest

- 🔗 **URL-Based Detection**  
  Extracts article text directly from news URLs using web scraping.

- 🧹 **Smart Preprocessing**  
  Automatically cleans and normalizes input text.

- 📊 **Performance Metrics**  
  Evaluates models using Accuracy, Precision, Recall, and F1 Score.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---------|------------|-------------|
| Backend | Python (Flask) | Web server and routing |
| Machine Learning | Scikit-Learn | Model training and prediction |
| Data Processing | Pandas | Dataset handling |

---

## 📂 Project Structure

```text
CSC3600Project/
├── app2.py              # Main Flask application
├── Fake.csv             # Fake news dataset
├── True.csv             # Real news dataset
└── README.md
⚙️ Installation & Setup
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/hannbella18/CSC3600Project.git
cd CSC3600Project
2️⃣ Install Dependencies
bash
Copy code
pip install flask pandas scikit-learn requests beautifulsoup4
3️⃣ Run the Application
bash
Copy code
python app2.py
The application will be available at:

cpp
Copy code
http://127.0.0.1:5000/
🧠 Model Details
🔹 Algorithms Used
Logistic Regression
Baseline model for binary classification.

Decision Tree
Captures non-linear data patterns.

Gradient Boosting
Ensemble technique that optimizes predictive errors.

Random Forest
Reduces overfitting by averaging multiple decision trees.

👤 Author
Hannbella
Computer Science Student @ UPM

GitHub: https://github.com/hannbella18

📄 License
This project is developed for academic purposes under the CSC3600 course.

