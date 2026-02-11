import streamlit as st
import pickle
import pandas as pd
import numpy as np
st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="centered")


import pickle

with open("ipl_pipeline.pkl", "rb") as f:
    model = pickle.load(f)





teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']
venues=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah',  'Mohali', 'Bengaluru']

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1629883061680-7f7d9e225a1e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown("""
    <style>
    .title {
        font-size:40px;
        font-weight:700;
        text-align:center;
        color:#1e3d59;
        padding: 10px;
    }
    .subtitle {
        text-align:center;
        font-size:20px;
        font-style:italic;
        color:#444;
        margin-top: -15px;
        margin-bottom: 20px;
    }
    .footer {
        text-align:center;
        font-size:13px;
        color:#888;
        margin-top: 50px;
    }
    </style>
    <div class="title">🏆 IPL Match Win Probability Predictor</div>
    <div class="subtitle">Strategize like a captain. Predict like a pro.</div>
""", unsafe_allow_html=True)
st.markdown("## 🧮 Match Situation")

batting_team = st.selectbox("🟢 Batting Team", teams)
bowling_team = st.selectbox("🔴 Bowling Team", [team for team in teams if team != batting_team])
venue = st.selectbox("🏟️ Match Venue", venues)
st.markdown("### ⏱️ Match Progress")

overs = st.slider("Overs Completed", 5.0, 20.0, step=0.1)
runs = st.number_input("Runs Scored", min_value=0)
wickets = st.slider("Wickets Lost", 0, 10)
target = st.number_input("Target Score", min_value=runs + 1)

if st.button("🔍 Predict Win Probabilities"):
    runs_left=target-runs
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=runs/overs
    rrr=(runs_left*6)/balls_left
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [venue],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    probabilities = model.predict_proba(input_df)[0]
    win_prob = round(probabilities[1] * 100, 2)
    lose_prob = round(probabilities[0] * 100, 2)

    st.markdown("## 📈 Win Probability Result")
    st.success(f"🏏 **{batting_team}** Win Probability: `{win_prob}%`")
    st.error(f"🛡️ **{bowling_team}** Win Probability: `{lose_prob}%`")
