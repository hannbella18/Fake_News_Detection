# 📰 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/AI-Scikit_Learn-orange)

**CSC3600 Project** is a Machine Learning-based web application developed to classify news articles as **Fake News** or **Not a Fake News (Real)**.

---

## 🚀 Key Features

* 🔍 **Multi-Model Prediction:** Simultaneously verifies news using 4 algorithms (Logistic Regression, Decision Tree, Gradient Boosting, Random Forest).
* 🔗 **URL-Based Detection:** Built-in web scraper extracts text directly from news links.
* 🧹 **Smart Preprocessing:** Automatically cleans input text.
* 📊 **Performance Metrics:** Evaluates models based on Accuracy, Precision, Recall, and F1 Score.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend** | Python (Flask) | Web server and API handling. |
| **Machine Learning** | Scikit-Learn | Model training and prediction. |
| **Data Processing** | Pandas | Handling True.csv and Fake.csv datasets. |

---

## 📂 Project Structure

```text
CSC3600Project/
├── app2.py              # Main Flask Application
├── Fake.csv             # Dataset (Fake News)
├── True.csv             # Dataset (Real News)
└── README.md

---

## ⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/hannbella18/CSC3600Project.git
cd CSC3600Project
2. Install Dependencies
pip install flask pandas scikit-learn requests beautifulsoup4
3. Run the Application
python app2.py
The application will launch at http://127.0.0.1:5000/

---

## 🧠 Model Details
🔹 Algorithms Used
Logistic Regression: Baseline model for binary classification.

Decision Tree: Captures non-linear data patterns.

Gradient Boosting: Ensemble technique that optimizes predictive errors.

Random Forest: Reduces overfitting by averaging multiple decision trees.

---

## 👤 Author
Hannbella

Computer Science Student @ UPM

GitHub Profile

---

## 📄 License
This project is developed for academic purposes under the CSC3600 course.


### 🧪 Final Check
After you paste this into VS Code:
1.  Look at the `git clone` part.
2.  Do you see the **three backticks** (` ``` `) above and below it?
3.  If yes, **Save** and **Push**!
4.  If no, type them in manually!
