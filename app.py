# Importing libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

# Loading the model and threshold
model_bundle = joblib.load('models/best_model.pkl')
model = model_bundle['model']
threshold = model_bundle['threshold']

# Title
st.title("All-NBA Selection Predictor")

# Description and note
st.write("""
This app predicts the probability that an NBA player will make an All-NBA team based on their season statistics.
""")
st.caption("Sliders are pre-filled with average player statistics.")


# Feature inputs
st.header("Enter Player Statistics")

def get_user_input():
    age = st.slider("Age", 18, 40, 26, step = 1)
    pts = st.slider("Points per Game", 0.0, 40.0, 11.0, step = .5)
    trb = st.slider("Total Rebounds per Game", 0.0, 20.0, 4.5, step = .5)
    ast = st.slider("Assists per Game", 0.0, 15.0, 2.5, step = .5)
    stl = st.slider("Steals per Game", 0.0, 5.0, 0.8, step = .1)
    blk = st.slider("Blocks per Game", 0.0, 5.0, 0.5, step = .1)
    x2p = st.slider("2PT FG%", 0.0, 1.0, 0.5, step = .01)
    x3p = st.slider("3PT FG%", 0.0, 1.0, 0.3, step = .01)
    ft = st.slider("FT%", 0.0, 1.0, 0.8, step = .01)
    x3p_ar = st.slider("3PT Attempt Rate", 0.0, 1.0, 0.4, step = .01)
    f_tr = st.slider("FT Rate", 0.0, 1.0, 0.3, step = .01)
    usg = st.slider("Usage%", 0.0, 50.0, 19.2, step = .1)
    orb = st.slider("Offensive Rebound%", 0.0, 20.0, 4.9, step = .1)
    drb = st.slider("Defensive Rebound%", 0.0, 40.0, 15.3, step = .1)
    ows = st.slider("Offensive Win Shares", -5.0, 10.0, 1.9, step = .1)
    dws = st.slider("Defensive Win Shares", -5.0, 10.0, 1.6, step = .1)
    obpm = st.slider("Offensive BPM", -10.0, 10.0, -0.3, step = .1)
    dbpm = st.slider("Defensive BPM", -10.0, 10.0, 0.0, step = .1)

    input_data = {
        'age': age,
        'pts_per_game': pts,
        'trb_per_game': trb,
        'ast_per_game': ast,
        'stl_per_game': stl,
        'blk_per_game': blk,
        'x2p_percent': x2p,
        'x3p_percent': x3p,
        'ft_percent': ft,
        'x3p_ar': x3p_ar,
        'f_tr': f_tr,
        'usg_percent': usg,
        'orb_percent': orb,
        'drb_percent': drb,
        'ows': ows,
        'dws': dws,
        'obpm': obpm,
        'dbpm': dbpm,
    }

    return pd.DataFrame([input_data])

# Get input
user_input = get_user_input()

# Prediction
if st.button("Predict"):
    prob = model.predict_proba(user_input)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class**: {'All-NBA' if prediction == 1 else 'Not All-NBA'}")
    st.write(f"**Probability of All-NBA**: {prob:.2%}")
    st.caption(f"Note: This model uses a tuned threshold of {threshold:.1%} to improve performance on imbalanced data.")

# Link to GitHub & Linkedin
st.markdown(
    """
    <hr style='margin-top:50px;'>
    <div style='text-align: center;'>
        Made by <b>Jack Ewings</b> &nbsp;•&nbsp; 
        <a href='https://github.com/jackewings/All-NBA-Prediction/tree/main' target='_blank'>View on GitHub</a> &nbsp;•&nbsp; 
        <a href='https://www.linkedin.com/in/jack-ewings-profile/' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)

