import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="centered")

# ---------------- LOAD MODEL ---------------- #
with open("ipl_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- DYNAMIC IMAGE SLIDER ---------------- #
st.markdown("""
<style>
.slider {
    width: 100%;
    height: 350px;
    overflow: hidden;
    position: relative;
    border-radius: 15px;
}

.slides {
    display: flex;
    width: 500%;
    animation: slide 20s infinite;
}

.slides img {
    width: 100%;
    height: 350px;
    object-fit: cover;
}

@keyframes slide {
    0% { margin-left: 0%; }
    20% { margin-left: 0%; }
    25% { margin-left: -100%; }
    45% { margin-left: -100%; }
    50% { margin-left: -200%; }
    70% { margin-left: -200%; }
    75% { margin-left: -300%; }
    95% { margin-left: -300%; }
    100% { margin-left: -400%; }
}

.quote {
    text-align: center;
    font-size: 22px;
    font-style: italic;
    color: white;
    margin-top: 15px;
    text-shadow: 1px 1px 6px black;
}
</style>

<div class="slider">
  <div class="slides">
    <img src="https://raw.githubusercontent.com/Satyam-Singh-x/ipl_pics/main/ipl-1.jpg">
    <img src="https://raw.githubusercontent.com/Satyam-Singh-x/ipl_pics/main/ipl-2.jpg">
    <img src="https://raw.githubusercontent.com/Satyam-Singh-x/ipl_pics/main/ipl-3.jpg">
    <img src="https://raw.githubusercontent.com/Satyam-Singh-x/ipl_pics/main/ipl-4.jpg">
    <img src="https://raw.githubusercontent.com/Satyam-Singh-x/ipl_pics/main/ipl-5.jpg">
  </div>
</div>

<div class="quote">
    “In cricket, every ball changes the game. In data, every feature changes the prediction.”
</div>
""", unsafe_allow_html=True)

# ---------------- BACKGROUND STYLING ---------------- #
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1508098682722-e99c643e7f0b?auto=format&fit=crop&w=1350&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.title {
    font-size:40px;
    font-weight:800;
    text-align:center;
    color:white;
    padding: 10px;
    text-shadow: 2px 2px 8px black;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:white;
    margin-bottom: 30px;
    text-shadow: 1px 1px 6px black;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏆 IPL Match Win Probability Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Strategize like a captain. Predict like a data scientist.</div>', unsafe_allow_html=True)

# ---------------- TEAMS & VENUES ---------------- #
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

venues = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

st.markdown("## 🧮 Match Situation")

batting_team = st.selectbox("🟢 Batting Team", teams)
bowling_team = st.selectbox("🔴 Bowling Team", [team for team in teams if team != batting_team])
venue = st.selectbox("🏟️ Match Venue", venues)

st.markdown("### ⏱️ Match Progress")

# ----------- CORRECT CRICKET OVERS LOGIC ----------- #
col1, col2 = st.columns(2)

with col1:
    over = st.number_input("Overs", min_value=0, max_value=19, step=1)

with col2:
    ball = st.number_input("Balls (0-5)", min_value=0, max_value=5, step=1)

overs = over + (ball / 6)

runs = st.number_input("Runs Scored", min_value=0)
wickets_lost = st.slider("Wickets Lost", 0, 9)
target = st.number_input("Target Score", min_value=runs + 1)

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Predict Win Probability"):

    runs_left = target - runs
    balls_left = 120 - (over * 6 + ball)
    wickets_remaining = 10 - wickets_lost
    crr = runs / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [venue],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    probabilities = model.predict_proba(input_df)[0]

    win_prob = round(probabilities[1] * 100, 2)
    lose_prob = round(probabilities[0] * 100, 2)

    st.markdown("## 📈 Match Prediction")

    col3, col4 = st.columns(2)

    with col3:
        st.success(f"🏏 {batting_team}")
        st.metric("Win Probability", f"{win_prob}%")

    with col4:
        st.error(f"🛡️ {bowling_team}")
        st.metric("Win Probability", f"{lose_prob}%")

    st.progress(int(win_prob))
