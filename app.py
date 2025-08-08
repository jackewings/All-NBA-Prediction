import streamlit as st
import pandas as pd
import joblib

# Load model and threshold dict
model_bundle = joblib.load('models/best_model.pkl')
pipeline = model_bundle['model']
threshold = float(model_bundle['threshold'])

# Load player stats
@st.cache_data
def load_player_stats():
    df = pd.read_csv('data/processed/player_stats_all_nba.csv')
    return df

player_stats_df = load_player_stats()

# Define feature list and means
feature_means = {
    'age': 26,
    'pts_per_game': 11.0,
    'trb_per_game': 4.5,
    'ast_per_game': 2.5,
    'stl_per_game': 0.8,
    'blk_per_game': 0.5,
    'x2p_percent': 0.5,
    'x3p_percent': 0.3,
    'ft_percent': 0.8,
    'x3p_ar': 0.4,
    'f_tr': 0.3,
    'usg_percent': 19.2,
    'orb_percent': 4.9,
    'drb_percent': 15.3,
    'ows': 1.9,
    'dws': 1.6,
    'obpm': -0.3,
    'dbpm': 0.0
}

# Define feature slider ranges
feature_ranges = {
    'age': (18, 40),
    'pts_per_game': (0.0, 35.0),
    'trb_per_game': (0.0, 15.0),
    'ast_per_game': (0.0, 12.0),
    'stl_per_game': (0.0, 5.0),
    'blk_per_game': (0.0, 5.0),
    'x2p_percent': (0.3, 0.75),
    'x3p_percent': (0.2, 0.5),
    'ft_percent': (0.5, 1.0),
    'x3p_ar': (0.0, 1.0),
    'f_tr': (0.0, 1.0),
    'usg_percent': (5.0, 40.0),
    'orb_percent': (0.0, 15.0),
    'drb_percent': (0.0, 30.0),
    'ows': (-2.0, 10.0),
    'dws': (-2.0, 10.0),
    'obpm': (-5.0, 10.0),
    'dbpm': (-5.0, 10.0)
}

# Customizing feature steps
feature_steps = {
    'age': 1.0,
    'pts_per_game': 0.1,
    'trb_per_game': 0.1,
    'ast_per_game': 0.1,
    'stl_per_game': 0.1,
    'blk_per_game': 0.1,
    'x2p_percent': 0.01,
    'x3p_percent': 0.01,
    'ft_percent': 0.01,
    'x3p_ar': 0.01,
    'f_tr': 0.01,
    'usg_percent': 0.1,
    'orb_percent': 0.1,
    'drb_percent': 0.1,
    'ows': 0.1,
    'dws': 0.1,
    'obpm': 0.1,
    'dbpm': 0.1
}

# Mapping for intuitive display names
feature_display_names = {
    'age': 'Age',
    'pts_per_game': 'Points Per Game',
    'trb_per_game': 'Rebounds Per Game',
    'ast_per_game': 'Assists Per Game',
    'stl_per_game': 'Steals Per Game',
    'blk_per_game': 'Blocks Per Game',
    'x2p_percent': '2P FG%',
    'x3p_percent': '3P FG%',
    'ft_percent': 'Free Throw %',
    'x3p_ar': '3P Attempt Rate',
    'f_tr': 'Free Throw Rate',
    'usg_percent': 'Usage %',
    'orb_percent': 'Offensive Rebound %',
    'drb_percent': 'Defensive Rebound %',
    'ows': 'Offensive Win Shares',
    'dws': 'Defensive Win Shares',
    'obpm': 'Offensive Box Plus/Minus',
    'dbpm': 'Defensive Box Plus/Minus'
}

st.title('All-NBA Prediction Tool')

st.markdown(
    '*If no player is selected or feature is left blank, average player values will be used automatically.*'
)

# Player autofill section
st.header('Optional: Autofill with Player Stats')
player_names = player_stats_df['player'].unique()
selected_player = st.selectbox('Select a player (optional):', [''] + list(player_names))

selected_season = None
player_row = None
if selected_player:
    filtered = player_stats_df[player_stats_df['player'] == selected_player]
    seasons = filtered['season'].unique()
    season_labels = {year: f'{year - 1}–{str(year)[-2:]}' for year in seasons}
    selected_season_label = st.selectbox(
    'Select a season:',
    [season_labels[year] for year in seasons]
    )
    selected_season = [year for year, label in season_labels.items() if label == selected_season_label][0]

    if selected_season:
        player_row = filtered[filtered['season'] == selected_season]
        st.success(f'Loaded stats for {selected_player} ({selected_season})')

# Feature input section
st.header('Choose Features to Adjust')

select_all = st.checkbox('Select All Features', value = False)

user_inputs = {}
for feature, mean_value in feature_means.items():
    col1, col2 = st.columns([1, 3])
    with col1:
        display_name = feature_display_names[feature]
        use_feature = st.checkbox(display_name, value = select_all, key = feature)
    if use_feature:
        with col2:
            default = (
                float(player_row[feature]) if player_row is not None and feature in player_row.columns
                else mean_value
            )
            min_val, max_val = feature_ranges[feature]
            user_input = st.slider(
                f'{display_name}',
                min_value = float(min_val),
                max_value = float(max_val),
                value = float(default),
                step = feature_steps.get(feature, 0.01),
                key = f'{feature}_input'
            )
            user_inputs[feature] = user_input
    else:
        if player_row is not None and feature in player_row.columns:
            user_inputs[feature] = float(player_row[feature])
        else:
            user_inputs[feature] = mean_value

# Prediction button
if st.button('Predict All-NBA Probability'):
    input_df = pd.DataFrame([user_inputs])
    prediction_proba = pipeline.predict_proba(input_df)[0][1]
    prediction_label = int(prediction_proba >= threshold)

    emoji = '✅' if prediction_label == 1 else '❌'
    label_text = 'All-NBA' if prediction_label == 1 else 'Not All-NBA'

    st.markdown(
        f"<h3 style='color:white; font-weight:bold;'>Prediction: {label_text} {emoji}</h3>",
        unsafe_allow_html = True
    )
    st.markdown(
        f"<h3 style='color:white; font-weight:bold;'>Probability of Making All-NBA: {prediction_proba:.2%}</h3>",
        unsafe_allow_html = True
    )
    st.progress(prediction_proba)

    st.markdown(
        '<hr style="margin-top: 30px; margin-bottom: 10px;">',
        unsafe_allow_html = True
    )
    st.caption(
        'Note: The prediction threshold for this model is **0.921** to adjust for class imbalance in the dataset.'
    )

st.markdown("---")
st.markdown(
    "👋 Connect with me on [LinkedIn](https://www.linkedin.com/in/jack-ewings-profile/) | "
    "[GitHub](https://github.com/jackewings)"
)










