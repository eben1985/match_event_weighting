import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# df_2024_SR = pd.read_csv('src/2024_SR.csv')
df_2024_SR_by_position = pd.read_csv('src/2024_SR_by_position.csv')

kick_errors = -0.3
rucks_won = 0.5
rucks_lost = -0.3
linebreaks = 0.6
tries = 1.4
supported_breaks = 0.3
defenders_beaten = 0.3
jackals_success = 0.4
intercepts = 0.3
tackles_made = 0.3
tackles_missed = -0.4
offloads = 0.3
carries = 0.2
carry_metres = 0.4
carries_dominant = 0.4
turnovers_conceded = -0.5
penalties_conceded = -0.6
carry_metres_post_contact = 0.3
ruck_arrivals_attack_first2 = 0.1
ruck_arrivals_defence_first2 = 0.1

st.header("Player score")
st.write("___________")
with st.sidebar:
    st.write("Adjust the weighting of each feature to see how it affects the player score.")
    kick_errors = st.slider(
        label="Kick Errors",
        min_value=-1.0,
        max_value=1.0,
        value=kick_errors,
        step=0.1
    )

    rucks_won = st.slider(
        label="Rucks Won",
        min_value=-1.0,
        max_value=1.0,
        value=rucks_won,
        step=0.1
    )

    rucks_lost = st.slider(
        label="Rucks Lost",
        min_value=-1.0,
        max_value=1.0,
        value=rucks_lost,
        step=0.1
    )

    linebreaks = st.slider(
        label="Linebreaks",
        min_value=-1.0,
        max_value=1.0,
        value=linebreaks,
        step=0.1
    )

    tries = st.slider(
        label="Tries",
        min_value=-1.0,
        max_value=2.0,
        value=tries,
        step=0.1
    )

    supported_breaks = st.slider(
        label="Supported Breaks",
        min_value=-1.0,
        max_value=1.0,
        value=supported_breaks,
        step=0.1
    )

    defenders_beaten = st.slider(
        label="Defenders Beaten",
        min_value=-1.0,
        max_value=1.0,
        value=defenders_beaten,
        step=0.1
    )

    jackals_success = st.slider(
        label="Jackals Success",
        min_value=-1.0,
        max_value=1.0,
        value=jackals_success,
        step=0.1
    )

    intercepts = st.slider(
        label="Intercepts",
        min_value=-1.0,
        max_value=1.0,
        value=intercepts,
        step=0.1
    )

    tackles_made = st.slider(
        label="Tackles Made",
        min_value=-1.0,
        max_value=1.0,
        value=tackles_made,
        step=0.1
    )

    tackles_missed = st.slider(
        label="Tackles Missed",
        min_value=-1.0,
        max_value=1.0,
        value=tackles_missed,
        step=0.1
    )

    offloads = st.slider(
        label="Offloads",
        min_value=-1.0,
        max_value=1.0,
        value=offloads,
        step=0.1
    )

    carries = st.slider(
        label="Carries",
        min_value=-1.0,
        max_value=1.0,
        value=carries,
        step=0.1
    )

    carry_metres = st.slider(
        label="Carry Metres",
        min_value=-1.0,
        max_value=1.0,
        value=carry_metres,
        step=0.1
    )

    carries_dominant = st.slider(
        label="Carries Dominant",
        min_value=-1.0,
        max_value=1.0,
        value=carries_dominant,
        step=0.1
    )

    turnovers_conceded = st.slider(
        label="Turnovers Conceded",
        min_value=-1.0,
        max_value=1.0,
        value=turnovers_conceded,
        step=0.1
    )

    penalties_conceded = st.slider(
        label="Penalties Conceded",
        min_value=-1.0,
        max_value=1.0,
        value=penalties_conceded,
        step=0.1
    )

    carry_metres_post_contact = st.slider(
        label="Carry Metres Post Contact",
        min_value=-1.0,
        max_value=1.0,
        value=carry_metres_post_contact,
        step=0.1
    )

    ruck_arrivals_attack_first2 = st.slider(
        label="Ruck Arrivals Attack First2",
        min_value=-1.0,
        max_value=1.0,
        value=ruck_arrivals_attack_first2,
        step=0.1
    )

    ruck_arrivals_defence_first2 = st.slider(
        label="Ruck Arrivals Defence First2",
        min_value=-1.0,
        max_value=1.0,
        value=ruck_arrivals_defence_first2,
        step=0.1
    )
    


slider_variables = {
    "kick_errors": kick_errors,
    "rucks_won": rucks_won,
    "rucks_lost": rucks_lost,
    "linebreaks": linebreaks,
    "tries": tries,
    "supported_breaks": supported_breaks,
    "defenders_beaten": defenders_beaten,
    "jackals_success": jackals_success,
    "intercepts": intercepts,
    "tackles_made": tackles_made,
    "tackles_missed": tackles_missed,
    "offloads": offloads,
    "carries": carries,
    "carry_metres": carry_metres,
    "carries_dominant": carries_dominant,
    "turnovers_conceded": turnovers_conceded,
    "penalties_conceded": penalties_conceded,
    "carry_metres_post_contact": carry_metres_post_contact,
    "ruck_arrivals_attack_first2": ruck_arrivals_attack_first2,
    "ruck_arrivals_defence_first2": ruck_arrivals_defence_first2
}

fig = px.bar(
    x=list(slider_variables.keys()),
    y=list(slider_variables.values()),
    labels={'x': 'Feature', 'y': 'Weighting'},
    title='Weighting Values'
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

df_2024_SR_by_position.dropna(inplace=True)
# drop all rows where position_id is 16
df_2024_SR_by_position = df_2024_SR_by_position[df_2024_SR_by_position['position_id'] != 16]

df_2024_SR_by_position['total_tries_m'] = (df_2024_SR_by_position['total_tries'] * tries).round(2)
df_2024_SR_by_position['total_linebreaks_m'] = (df_2024_SR_by_position['total_linebreaks'] * linebreaks).round(2)
df_2024_SR_by_position['total_defenders_beaten_m'] = (df_2024_SR_by_position['total_defenders_beaten'] * defenders_beaten).round(2)
df_2024_SR_by_position['total_jackals_success_m'] = (df_2024_SR_by_position['total_jackals_success'] * jackals_success).round(2)
df_2024_SR_by_position['total_tackles_made_m'] = (df_2024_SR_by_position['total_tackles_made'] * tackles_made).round(2)
df_2024_SR_by_position['total_intercepts_m'] = (df_2024_SR_by_position['total_intercepts'] * intercepts).round(2)
df_2024_SR_by_position['total_tackles_missed_m'] = (df_2024_SR_by_position['total_tackles_missed'] * tackles_missed).round(2)
df_2024_SR_by_position['total_penalties_conceded_m'] = (df_2024_SR_by_position['total_penalties_conceded'] * penalties_conceded).round(2)
df_2024_SR_by_position['total_carry_metres_m'] = (df_2024_SR_by_position['total_carry_metres'] * carry_metres / 10).round(2)
df_2024_SR_by_position['total_carries_m'] = (df_2024_SR_by_position['total_carries'] * carries).round(2)
df_2024_SR_by_position['total_carries_dominant_m'] = (df_2024_SR_by_position['total_carries_dominant'] * carries_dominant).round(2)
df_2024_SR_by_position['total_supported_breaks_m'] = (df_2024_SR_by_position['total_supported_breaks'] * supported_breaks).round(2)
df_2024_SR_by_position['total_offloads_m'] = (df_2024_SR_by_position['total_offloads'] * offloads).round(2)
df_2024_SR_by_position['total_turnovers_conceded_m'] = (df_2024_SR_by_position['total_turnovers_conceded'] * turnovers_conceded)

df_2024_SR_by_position['total_kick_errors_m'] = (df_2024_SR_by_position['total_kick_errors'] * kick_errors).round(2)
df_2024_SR_by_position['total_rucks_won_m'] = (df_2024_SR_by_position['total_rucks_won'] * rucks_won).round(2)
df_2024_SR_by_position['total_rucks_lost_m'] = (df_2024_SR_by_position['total_rucks_lost'] * rucks_lost).round(2)
df_2024_SR_by_position['carry_metres_post_contact_m'] = (df_2024_SR_by_position['total_carry_metres_post_contact'] * carry_metres_post_contact).round(2)
df_2024_SR_by_position['ruck_arrivals_attack_first2_m'] = (df_2024_SR_by_position['total_ruck_arrivals_attack_first2'] * ruck_arrivals_attack_first2).round(2)
df_2024_SR_by_position['ruck_arrivals_defence_first2_m'] = (df_2024_SR_by_position['total_ruck_arrivals_defence_first2'] * ruck_arrivals_defence_first2).round(2)

#count these columns to create a total_score column
df_2024_SR_by_position['total_score'] = df_2024_SR_by_position['total_tries_m'] + \
    df_2024_SR_by_position['total_linebreaks_m'] + df_2024_SR_by_position['total_defenders_beaten_m'] + \
        df_2024_SR_by_position['total_jackals_success_m'] + df_2024_SR_by_position['total_tackles_made_m'] + \
            df_2024_SR_by_position['total_intercepts_m'] + df_2024_SR_by_position['total_tackles_missed_m'] + df_2024_SR_by_position['total_penalties_conceded_m'] + \
                df_2024_SR_by_position['total_carry_metres_m'] + df_2024_SR_by_position['total_carries_m'] + \
                    df_2024_SR_by_position['total_carries_dominant_m'] + df_2024_SR_by_position['total_supported_breaks_m'] + \
                        df_2024_SR_by_position['total_offloads_m'] + df_2024_SR_by_position['total_turnovers_conceded_m'] + \
                            df_2024_SR_by_position['carry_metres_post_contact_m'] + df_2024_SR_by_position['ruck_arrivals_attack_first2_m'] + df_2024_SR_by_position['ruck_arrivals_defence_first2_m'] +\
                                df_2024_SR_by_position['total_kick_errors_m'] + df_2024_SR_by_position['total_rucks_won_m'] + df_2024_SR_by_position['total_rucks_lost_m']

#count these columns to create a total_score_defence column
df_2024_SR_by_position['total_score_defence'] = df_2024_SR_by_position['total_jackals_success_m'] + df_2024_SR_by_position['total_tackles_made_m'] + \
    df_2024_SR_by_position['total_intercepts_m'] + df_2024_SR_by_position['total_tackles_missed_m'] + df_2024_SR_by_position['ruck_arrivals_defence_first2_m']

#count these columns to create a total_score_attack column
df_2024_SR_by_position['total_score_attack'] = df_2024_SR_by_position['total_tries_m'] + \
    df_2024_SR_by_position['total_linebreaks_m'] + df_2024_SR_by_position['total_defenders_beaten_m'] + \
        df_2024_SR_by_position['total_carry_metres_m'] + df_2024_SR_by_position['total_carries_m'] + \
            df_2024_SR_by_position['total_carries_dominant_m'] + df_2024_SR_by_position['total_supported_breaks_m'] + \
                df_2024_SR_by_position['carry_metres_post_contact_m'] + df_2024_SR_by_position['ruck_arrivals_attack_first2_m'] + \
                    df_2024_SR_by_position['total_rucks_won_m'] + df_2024_SR_by_position['total_rucks_lost_m']

# new column that divde the total_score by the number of BIP minutes played
df_2024_SR_by_position['total_score_per_BIP'] = (df_2024_SR_by_position['total_score'] / df_2024_SR_by_position['total_ballinplay_minutes']).round(2)
df_2024_SR_by_position['total_score_defence_per_defmin'] = (df_2024_SR_by_position['total_score_defence'] / df_2024_SR_by_position['total_def_minutes']).round(2)
df_2024_SR_by_position['total_score_attack_per_attmin'] = (df_2024_SR_by_position['total_score_attack'] / df_2024_SR_by_position['total_att_minutes']).round(2)

def group_and_sum(df):
    """
    Groups the DataFrame by player_mid and player_name, and sums the specified columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The grouped and summed DataFrame.
    """
    # List of columns to sum
    columns_to_sum = [
    'total_minutes_played',
    'total_ballinplay_minutes',
    'total_att_minutes',
    'total_def_minutes',
    'total_tries', 
    'total_linebreaks', 
    'total_defenders_beaten', 
    'total_jackals_success', 
    'total_tackles_made', 
    'total_intercepts', 
    'total_tackles_missed', 
    'total_penalties_conceded', 
    'total_carry_metres', 
    'total_carries', 
    'total_carries_dominant', 
    'total_supported_breaks', 
    'total_offloads', 
    'total_turnovers_conceded',
    'total_score',
    'total_score_per_BIP',
    'total_score_defence',
    'total_score_attack',
    'total_score_defence_per_defmin',
    'total_score_attack_per_attmin',
    'total_kick_errors', 
    'total_rucks_won', 
    'total_rucks_lost', 
    'total_carry_metres_post_contact',
    'total_ruck_arrivals_attack_first2', 
    'total_ruck_arrivals_defence_first2'
]

    # Group by player_mid and player_name and sum the specified columns
    grouped_df = df.groupby(['player_mid', 'player_name','team_name'])[columns_to_sum].sum().reset_index()
    return grouped_df

grouped_df = group_and_sum(df_2024_SR_by_position)

teams_list = grouped_df['team_name'].unique()
selected_teams = st.multiselect('Select a team', teams_list,teams_list)

#filter to only show selected teams in the teams list
# have all the teams in the list as a starting point
grouped_df = grouped_df[grouped_df['team_name'].isin(selected_teams)]


# display a dataframe with the player_name and the total_score
grouped_df = grouped_df[['player_name', 'total_score', 'total_score_per_BIP']].sort_values('total_score', ascending=False)

# Reset the index to drop the default index
grouped_df = grouped_df.reset_index(drop=True)

# Display the dataframe without the index column and using the container width
st.dataframe(grouped_df, use_container_width=True)


### by position
st.subheader('By Position')
#select position
positions_list = sorted(df_2024_SR_by_position['position_id'].unique())

selected_position = st.radio('Select a position', positions_list, horizontal=True)
#filter by selected postion
df_position = df_2024_SR_by_position[df_2024_SR_by_position['position_id'] == selected_position]
df_position = df_position[df_position['team_name'].isin(selected_teams)]

df_position = df_position[['player_name', 'total_score', 'total_score_per_BIP']].sort_values('total_score', ascending=False)
# Convert the index to range and drop the default index
df_position = df_position.reset_index(drop=True)

st.dataframe(df_position, use_container_width=True)

col3, col4 = st.columns(2)
df_position_filtered = df_2024_SR_by_position[(df_2024_SR_by_position['position_id'] == selected_position) & (df_2024_SR_by_position['team_name'].isin(selected_teams))]
with col3:
    st.subheader('Attacking score')
    df_position_attack_score = df_position_filtered[['player_name', 'total_score_attack', 'total_score_attack_per_attmin']].sort_values('total_score_attack', ascending=False)
    st.dataframe(df_position_attack_score)
with col4:
    st.subheader('Defence score')
    df_position_defence_score = df_position_filtered[['player_name', 'total_score_defence', 'total_score_defence_per_defmin']].sort_values('total_score_defence', ascending=False)
    st.dataframe(df_position_defence_score)

st.dataframe(df_position_filtered)


# use df_position_filtered to make a scatter plot with plotly having on the x-axis the total_score_attack_per_attmin and the total_score_defence_per_defmin on the y-axis show player-name on the plot
fig = px.scatter(
    df_position_filtered, 
    x='total_score_attack_per_attmin', 
    y='total_score_defence_per_defmin', 
    text='player_name', 
    hover_name='player_name',
    title='Player score by position',
)
# Update the text position to be above the markers
fig.update_traces(textposition='top center')

# Adjust the chart size
fig.update_layout(
    # width=600,  # Adjust the width as needed
    height=700,  # Adjust the height as needed
    plot_bgcolor='#f5f5f5',  # Set the plot background color to light grey
    xaxis_title='Attack Score per Minute',  # Update the x-axis title
    yaxis_title='Defence Score per Minute'  # Update the y-axis title
)

# Display the Plotly chart in Streamlit with full container width
st.plotly_chart(fig, use_container_width=True)


    
    
st.write("___________")

with st.expander("Elo scoring"):
    #add a toggle button in streamlit
    on = st.toggle("Include match difficulty scoring")

    if on:
        st.write("Including Elo Scoring...")
        grouped_df['random_multiplier'] = np.random.randint(0, 3, size=len(grouped_df))
        grouped_df['total_score_elo'] = grouped_df['total_score'] * grouped_df['random_multiplier']

        # Selecting and sorting the desired columns
        result_df = grouped_df[['player_name', 'total_score', 'total_score_per_BIP', 'total_score_elo']].sort_values('total_score_elo', ascending=False)

        
        st.dataframe(result_df[['player_name', 'total_score_elo','total_score_per_BIP']].sort_values('total_score_elo', ascending=False))

st.write("___________")

with st.expander("Player score by match"):

    player = st.selectbox('Select a player', grouped_df['player_name'].unique())
    if player is not None:
        st.write(f"Player selected: {player}")
        
        # Create 10 random numbers between 0 and 1 to put on the y axis and numbers 1 to 10 on the x axis in a Plotly line chart
        random_numbers = np.random.rand(10)
        x = np.arange(1, 11)
        
        # Create a DataFrame to use with Plotly Express
        data = pd.DataFrame({
            'Games': x,
            'Player Score per Minute': random_numbers
        })
        
        # Create the scatter plot with a trend line
        fig = px.scatter(data, x='Games', y='Player Score per Minute', title=f"{player}", trendline='ols', labels={'x': 'Games', 'y': 'Player Score per Minute'})
        
        # Add a line trace for the original data points
        fig.add_scatter(x=x, y=random_numbers, mode='lines', name='Player Score per Minute by match')
        
        # Update the trend line to be smaller and red
        fig.update_traces(selector=dict(name='Trend line'), line=dict(color='red', width=2))
        
        st.plotly_chart(fig)
        
        #sql query to get players: df_2024_SR_by_position
        '''
        SELECT 
        player_mid,
        player_name, 
        position_id,
        team_name,
    EXTRACT(YEAR FROM fixture_datetime) as fixture_year, -- New column for the year
        sum(carry_metres_post_contact) as total_carry_metres_post_contact,
        sum(ballinplay_minutes) as total_ballinplay_minutes,
        sum(att_minutes) as total_att_minutes,
        sum(def_minutes) as total_def_minutes,
        sum(kick_errors) as total_kick_errors,
        sum(ruck_arrivals_attack_first2) as total_ruck_arrivals_attack_first2,
        sum(ruck_arrivals_defence_first2) as total_⁠⁠ruck_arrivals_defence_first2,
        SUM(minutes_played) AS total_minutes_played, 
        SUM(tries) AS total_tries, 
        SUM(linebreaks) AS total_linebreaks,
        SUM(supported_break) AS total_supported_breaks,
        SUM(defenders_beaten) AS total_defenders_beaten,
        SUM(jackals_success) AS total_jackals_success,
        SUM(intercepts) AS total_intercepts,
        SUM(tackles_made) AS total_tackles_made,
        SUM(tackles_missed) AS total_tackles_missed,
        SUM(offloads) AS total_offloads,
        SUM(carries) AS total_carries,
        SUM(carry_metres) AS total_carry_metres,
        SUM(carries_dominant) AS total_carries_dominant,
        SUM(turnovers_conceded) AS total_turnovers_conceded,
        SUM(penalties_conceded) AS total_penalties_conceded, 
        SUM(rucks_won) as total_rucks_won,
        SUM(rucks_lost) as total_rucks_lost
    FROM 
        bal.tab_match_player_stats_v2
    WHERE 
        fixture_year = 2024
        AND competition = 'Super Rugby Pacific'
    GROUP BY 
        player_mid, 
        player_name, 
        position_id,
        team_name,
        EXTRACT(YEAR FROM fixture_datetime)
    order by player_name;
'''