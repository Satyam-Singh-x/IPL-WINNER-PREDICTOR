🏏 IPL Match Winner Predictor

Live demo: https://ipl-winner-predictor-bysatyam.streamlit.app/

Welcome to the IPL Match Winner Predictor — a machine learning-based web application that predicts the outcome of Indian Premier League (IPL) matches in real time using match context and team performance.

Built with Logistic Regression and powered by Streamlit, this interactive app gives cricket fans and data enthusiasts a glimpse into the power of predictive analytics on the field.

📌 Features

✅ Predicts match-winning probabilities dynamically

✅ Real-time input for live match conditions

✅ Stylish, competitive UI for an immersive cricket experience

✅ Fully deployed and shareable via Streamlit Cloud

🧠 Model Overview

Algorithm: Logistic Regression

Training Dataset: IPL ball-by-ball and match-level data (matches.csv & deliveries.csv)

Key Features Used:

Current score

Overs and wickets

Batting and bowling teams

Venue

Target score

Run rate and required run rate

🛠️ Technologies Used

Python 3.10

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Pickle

Kaggle

🔧 Setup Instructions

Clone the Repository
bash

git clone https://github.com/Satyam-Singh-x/IPL-match-predictor.git

cd IPL-match-predictor

Create Virtual Environment (optional but recommended)
python -m venv .venv

source .venv/bin/activate # For Linux/Mac

.venv\Scripts\activate # For Windows

Install Dependencies bash
pip install -r requirements.txt

Run Locally
bash

streamlit run app.py

🗂️ Project Structure

python

📦 ipl-match-predictor/ ├── app.py # Streamlit app

├── pipe.pkl # Trained ML model

├── matches.csv # Match-level data

├── deliveries.csv # Ball-by-ball data

├── ipl-match-predictor.ipynb #Complete EDA ,feature engineering and model building

├── requirements.txt # Dependencies

└── runtime.txt # Python version (for Streamlit Cloud)

⚡ Deployment (Streamlit Cloud)

Push your code to a public GitHub repo

Include requirements.txt and runtime.txt

Deploy via Streamlit Cloud

Done! 🎯

📢 Credits

Data Source: Kaggle IPL Dataset

For cricket fans, analysts, and dreamers of the game.

📬 Contact

Satyam Singh

Feel free to reach out for collaborations, ideas, or feedback:

singhsatyam.0912@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/satyam-singh-61152a334
