import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from streamlit_functions import sidebar, weighting_bar_graph


# df_2024_SR = pd.read_csv('src/2024_SR.csv')
df_2024_SR_by_fixture = pd.read_csv('src/2024_SR_grouped_fixture_true.csv')



st.header("Player score")
st.write("___________")
#sidebar
kick_errors,rucks_won,rucks_lost,linebreaks,tries,supported_breaks,\
defenders_beaten,jackals_success,intercepts,tackles_made,tackles_missed,\
offloads,carries,carry_metres,carries_dominant,turnovers_conceded,penalties_conceded,\
carry_metres_post_contact,ruck_arrivals_attack_first2,ruck_arrivals_defence_first2 = sidebar()

#weighitings bar graph
fig_bar_weights = weighting_bar_graph(kick_errors,rucks_won,rucks_lost,linebreaks,tries,supported_breaks,\
defenders_beaten,jackals_success,intercepts,tackles_made,tackles_missed,\
offloads,carries,carry_metres,carries_dominant,turnovers_conceded,penalties_conceded,\
carry_metres_post_contact,ruck_arrivals_attack_first2,ruck_arrivals_defence_first2)
st.plotly_chart(fig_bar_weights, use_container_width=True)

#clean the data from nulls and position_id 16
df_2024_SR_by_fixture.dropna(inplace=True)
df_2024_SR_by_fixture = df_2024_SR_by_fixture[df_2024_SR_by_fixture['position_id'] != 16]

#dropdown with a list of fixture_teams
fixture_teams = df_2024_SR_by_fixture['fixture_teams'].unique()
#add a item to the list called 'All matches'
fixture_teams = np.insert(fixture_teams, 0, 'All Matches')
selected_fixture_team = st.selectbox('Select a team', fixture_teams)
if selected_fixture_team == 'All Matches':
    # set selected_fixture_team =  to the new name of the dataframe df_2024_SR_by_fixture_filtered_match'
    df_2024_SR_by_fixture_filtered_match = df_2024_SR_by_fixture
    teams_list = df_2024_SR_by_fixture_filtered_match['team_name'].unique()
    selected_teams = st.multiselect('Select a team', teams_list,teams_list)

    #filter to only show selected teams in the teams list
    # have all the teams in the list as a starting point
    df_2024_SR_by_fixture_filtered_match = df_2024_SR_by_fixture_filtered_match[df_2024_SR_by_fixture_filtered_match['team_name'].isin(selected_teams)]
else:
    df_2024_SR_by_fixture_filtered_match = df_2024_SR_by_fixture[df_2024_SR_by_fixture['fixture_teams'] == selected_fixture_team]
        
# #get columns that count the matches and sum the minutes so that they can be filtereed out before the calculation runs
# # Sum the total minutes played for each 'player_mid'
df_2024_SR_by_fixture_filtered_match['total_ballinplay_minutes_sum'] = df_2024_SR_by_fixture_filtered_match.groupby(['player_mid','position_id'])['total_ballinplay_minutes'].transform('sum')
# # Count the number of games each 'player_mid' has played
df_2024_SR_by_fixture_filtered_match['games_played'] = df_2024_SR_by_fixture_filtered_match.groupby(['player_mid','position_id'])['player_mid'].transform('count')
st.dataframe(df_2024_SR_by_fixture_filtered_match)

#get max sumed minutes played and games played
max_minutes_played = df_2024_SR_by_fixture_filtered_match['total_ballinplay_minutes_sum'].max()   
max_games_played = df_2024_SR_by_fixture_filtered_match['games_played'].max()
number_of_games_player_filter = 1

st.write("Filter by minutes played and games played")
col1, col2 = st.columns(2)
with col1:
    start_filter_value_minutes, end_filter_value_minutes = st.slider('Select the range of minutes played', 0, max_minutes_played, (0, max_minutes_played))
with col2:
    if selected_fixture_team == 'All Matches':
        number_of_games_player_filter = st.slider(label="Select the minimum number of games played",min_value=1,max_value=max_games_played,value=1,step=1)

# Filters the DataFrame based on the selected range of minutes played and more and inclusive ofgames played
df_2024_SR_filtered_by_games_and_minutes = df_2024_SR_by_fixture_filtered_match[(df_2024_SR_by_fixture_filtered_match['total_ballinplay_minutes_sum'] >= start_filter_value_minutes) & (df_2024_SR_by_fixture_filtered_match['total_ballinplay_minutes_sum'] <= end_filter_value_minutes) & (df_2024_SR_by_fixture_filtered_match['games_played'] >= number_of_games_player_filter)]

df_2024_SR_filtered_by_games_and_minutes['total_tries_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_tries'] * tries).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_linebreaks_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_linebreaks'] * linebreaks).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_defenders_beaten_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_defenders_beaten'] * defenders_beaten).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_jackals_success_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_jackals_success'] * jackals_success).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_tackles_made_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_tackles_made'] * tackles_made).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_intercepts_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_intercepts'] * intercepts).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_tackles_missed_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_tackles_missed'] * tackles_missed).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_penalties_conceded_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_penalties_conceded'] * penalties_conceded).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_carry_metres'] * carry_metres * 0.1).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_carries_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_carries'] * carries).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_carries_dominant_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_carries_dominant'] * carries_dominant).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_supported_breaks_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_supported_breaks'] * supported_breaks).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_offloads_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_offloads'] * offloads).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_turnovers_conceded_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_turnovers_conceded'] * turnovers_conceded)

df_2024_SR_filtered_by_games_and_minutes['total_kick_errors_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_kick_errors'] * kick_errors).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_rucks_won_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_rucks_won'] * rucks_won).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_rucks_lost_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_rucks_lost'] * rucks_lost).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_post_contact_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_post_contact'] * carry_metres_post_contact * 0.1).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_attack_first2_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_attack_first2'] * ruck_arrivals_attack_first2).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_defence_first2_m'] = (df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_defence_first2'] * ruck_arrivals_defence_first2).round(2)


#count these columns to create a total_score column
df_2024_SR_filtered_by_games_and_minutes['total_score'] = df_2024_SR_filtered_by_games_and_minutes['total_tries_m'] + \
    df_2024_SR_filtered_by_games_and_minutes['total_linebreaks_m'] + df_2024_SR_filtered_by_games_and_minutes['total_defenders_beaten_m'] + \
        df_2024_SR_filtered_by_games_and_minutes['total_jackals_success_m'] + df_2024_SR_filtered_by_games_and_minutes['total_tackles_made_m'] + \
            df_2024_SR_filtered_by_games_and_minutes['total_intercepts_m'] + df_2024_SR_filtered_by_games_and_minutes['total_tackles_missed_m'] + \
                df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_m'] + df_2024_SR_filtered_by_games_and_minutes['total_carries_m'] + \
                    df_2024_SR_filtered_by_games_and_minutes['total_carries_dominant_m'] + df_2024_SR_filtered_by_games_and_minutes['total_supported_breaks_m'] + \
                        df_2024_SR_filtered_by_games_and_minutes['total_offloads_m'] + df_2024_SR_filtered_by_games_and_minutes['total_turnovers_conceded_m'] + \
                            df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_post_contact_m'] + df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_attack_first2_m'] + df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_defence_first2_m'] +\
                                df_2024_SR_filtered_by_games_and_minutes['total_kick_errors_m'] + df_2024_SR_filtered_by_games_and_minutes['total_penalties_conceded_m'] +\
                                    df_2024_SR_filtered_by_games_and_minutes['total_rucks_won_m'] + df_2024_SR_filtered_by_games_and_minutes['total_rucks_lost_m']

# st.dataframe(df_2024_SR_filtered_by_games_and_minutes)

# #count these columns to create a total_score_defence column
df_2024_SR_filtered_by_games_and_minutes['total_score_defence'] = df_2024_SR_filtered_by_games_and_minutes['total_jackals_success_m'] + df_2024_SR_filtered_by_games_and_minutes['total_tackles_made_m'] + \
    df_2024_SR_filtered_by_games_and_minutes['total_intercepts_m'] + df_2024_SR_filtered_by_games_and_minutes['total_tackles_missed_m'] + df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_defence_first2_m']

#count these columns to create a total_score_attack column
df_2024_SR_filtered_by_games_and_minutes['total_score_attack'] = df_2024_SR_filtered_by_games_and_minutes['total_tries_m'] + \
    df_2024_SR_filtered_by_games_and_minutes['total_linebreaks_m'] + df_2024_SR_filtered_by_games_and_minutes['total_defenders_beaten_m'] + \
        df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_m'] + df_2024_SR_filtered_by_games_and_minutes['total_carries_m'] + \
            df_2024_SR_filtered_by_games_and_minutes['total_carries_dominant_m'] + df_2024_SR_filtered_by_games_and_minutes['total_supported_breaks_m'] + \
                df_2024_SR_filtered_by_games_and_minutes['total_carry_metres_post_contact_m'] + df_2024_SR_filtered_by_games_and_minutes['total_ruck_arrivals_attack_first2_m'] +\
                   df_2024_SR_filtered_by_games_and_minutes['total_rucks_won_m'] + df_2024_SR_filtered_by_games_and_minutes['total_rucks_lost_m']

# new column that divde the total_score by the number of BIP minutes played
df_2024_SR_filtered_by_games_and_minutes['total_score_per_BIP'] = (df_2024_SR_filtered_by_games_and_minutes['total_score'] / df_2024_SR_filtered_by_games_and_minutes['total_ballinplay_minutes']).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_score_defence_per_defmin'] = (df_2024_SR_filtered_by_games_and_minutes['total_score_defence'] / df_2024_SR_filtered_by_games_and_minutes['total_def_minutes']).round(2)
df_2024_SR_filtered_by_games_and_minutes['total_score_attack_per_attmin'] = (df_2024_SR_filtered_by_games_and_minutes['total_score_attack'] / df_2024_SR_filtered_by_games_and_minutes['total_att_minutes']).round(2)

st.dataframe(df_2024_SR_filtered_by_games_and_minutes)


# List of columns to sum
columns_to_sum = [
'total_minutes_played',
'total_ballinplay_minutes',
'total_att_minutes',
'total_def_minutes',
'total_tries_m', 
'total_linebreaks_m', 
'total_defenders_beaten_m', 
'total_jackals_success_m', 
'total_tackles_made_m', 
'total_intercepts_m', 
'total_tackles_missed_m', 
'total_penalties_conceded_m', 
'total_carry_metres_m', 
'total_carries_m', 
'total_carries_dominant_m', 
'total_supported_breaks_m', 
'total_offloads_m', 
'total_turnovers_conceded_m',
'total_score',
'total_score_per_BIP',
'total_score_defence',
'total_score_attack',
'total_score_defence_per_defmin',
'total_score_attack_per_attmin',
'total_kick_errors_m', 
'total_rucks_won_m', 
'total_rucks_lost_m', 
'total_carry_metres_post_contact_m',
'total_ruck_arrivals_attack_first2_m', 
'total_ruck_arrivals_defence_first2_m'
]
# Group by player_mid and player_name and sum the specified columns
# grouped_df = df_2024_SR_filtered_by_games_and_minutes.groupby(['player_mid', 'player_name','team_name'])[columns_to_sum].sum().reset_index()
# st.dataframe(grouped_df)

# teams_list = grouped_df['team_name'].unique()
# selected_teams = st.multiselect('Select a team', teams_list,teams_list)

# #filter to only show selected teams in the teams list
# # have all the teams in the list as a starting point
# grouped_df = grouped_df[grouped_df['team_name'].isin(selected_teams)]


# # display a dataframe with the player_name and the total_score
# grouped_df = grouped_df[['player_name', 'total_score', 'total_score_per_BIP']].sort_values('total_score', ascending=False)

# # Reset the index to drop the default index
# grouped_df = grouped_df.reset_index(drop=True)

# # Display the dataframe without the index column and using the container width
# st.dataframe(grouped_df, use_container_width=True)


# ### by position
st.subheader('By Position')
#select position
positions_list = sorted(df_2024_SR_filtered_by_games_and_minutes['position_id'].unique())

selected_position = st.radio('Select a position', positions_list, horizontal=True)
#filter by selected postion
df_position = df_2024_SR_filtered_by_games_and_minutes[df_2024_SR_filtered_by_games_and_minutes['position_id'] == selected_position]
df_position = df_position[df_position['team_name'].isin(selected_teams)]

grouped_position_df = df_position.groupby(['player_mid', 'player_name','team_name','position_id'])[columns_to_sum].sum().reset_index()
# interesting this should be done at a higher level and not just the sum or groupby of score_per_BIP
grouped_position_df['total_score_per_BIP_full'] = (grouped_position_df['total_score'] / grouped_position_df['total_ballinplay_minutes']).round(2)
grouped_position_df['total_score_attack_per_attmin_full'] = (grouped_position_df['total_score_attack'] / grouped_position_df['total_att_minutes']).round(2)
grouped_position_df['total_score_defence_per_defmin_full'] = (grouped_position_df['total_score_defence'] / grouped_position_df['total_def_minutes']).round(2)
st.write("grouped with full")
st.dataframe(grouped_position_df)

#make df smaller
grouped_position_df_totals = grouped_position_df[['player_name', 'total_score', 'total_score_per_BIP_full']].sort_values('total_score', ascending=False)
# Convert the index to range and drop the default index
grouped_position_df_totals = grouped_position_df_totals.reset_index(drop=True)

st.dataframe(grouped_position_df_totals, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader('Attacking score')
    df_position_attack_score = grouped_position_df[['player_name', 'total_score_attack', 'total_score_attack_per_attmin_full']].sort_values('total_score_attack', ascending=False)
    st.dataframe(df_position_attack_score)
with col4:
    st.subheader('Defence score')
    df_position_defence_score = grouped_position_df[['player_name', 'total_score_defence', 'total_score_defence_per_defmin_full']].sort_values('total_score_defence', ascending=False)
    st.dataframe(df_position_defence_score)

# st.dataframe(grouped_position_df)


# use df_position_filtered to make a scatter plot with plotly having on the x-axis the total_score_attack_per_attmin and the total_score_defence_per_defmin on the y-axis show player-name on the plot
fig = px.scatter(
    grouped_position_df, 
    x='total_score_attack_per_attmin_full', 
    y='total_score_defence_per_defmin_full', 
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


    
    
# st.write("___________")

# with st.expander("Elo scoring"):
#     #add a toggle button in streamlit
#     on = st.toggle("Include match difficulty scoring")

#     if on:
#         st.write("Including Elo Scoring...")
#         grouped_df['random_multiplier'] = np.random.randint(0, 3, size=len(grouped_df))
#         grouped_df['total_score_elo'] = grouped_df['total_score'] * grouped_df['random_multiplier']

#         # Selecting and sorting the desired columns
#         result_df = grouped_df[['player_name', 'total_score', 'total_score_per_BIP', 'total_score_elo']].sort_values('total_score_elo', ascending=False)

        
#         st.dataframe(result_df[['player_name', 'total_score_elo','total_score_per_BIP']].sort_values('total_score_elo', ascending=False))

# st.write("___________")

# with st.expander("Player score by match"):

#     player = st.selectbox('Select a player', grouped_df['player_name'].unique())
#     if player is not None:
#         st.write(f"Player selected: {player}")
        
#         # Create 10 random numbers between 0 and 1 to put on the y axis and numbers 1 to 10 on the x axis in a Plotly line chart
#         random_numbers = np.random.rand(10)
#         x = np.arange(1, 11)
        
#         # Create a DataFrame to use with Plotly Express
#         data = pd.DataFrame({
#             'Games': x,
#             'Player Score per Minute': random_numbers
#         })
        
#         # Create the scatter plot with a trend line
#         fig = px.scatter(data, x='Games', y='Player Score per Minute', title=f"{player}", trendline='ols', labels={'x': 'Games', 'y': 'Player Score per Minute'})
        
#         # Add a line trace for the original data points
#         fig.add_scatter(x=x, y=random_numbers, mode='lines', name='Player Score per Minute by match')
        
#         # Update the trend line to be smaller and red
#         fig.update_traces(selector=dict(name='Trend line'), line=dict(color='red', width=2))
        
#         st.plotly_chart(fig)
        
#sql query to get players: df_2024_SR_by_position
'''
        SELECT 
        fixture_mid,
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
        fixture_mid,
        player_mid, 
        player_name, 
        position_id,
        team_name,
        EXTRACT(YEAR FROM fixture_datetime)
    order by fixture_mid;
'''