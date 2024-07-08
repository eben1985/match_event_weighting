import streamlit as st
import plotly.express as px

#create sidebar
def sidebar():
    with st.sidebar:
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
        
        st.write("Adjust the weighting of each feature to see how it affects the player score.")
        kick_errors = st.slider(label="Kick Errors",min_value=-1.0,max_value=1.0,value=kick_errors,step=0.1)
        rucks_won = st.slider(label="Rucks Won",min_value=-1.0,max_value=1.0,value=rucks_won,step=0.1)
        rucks_lost = st.slider(label="Rucks Lost",min_value=-1.0,max_value=1.0,value=rucks_lost,step=0.1)
        linebreaks = st.slider(label="Linebreaks",min_value=-1.0,max_value=1.0,value=linebreaks,step=0.1)
        tries = st.slider(label="Tries",min_value=-1.0,max_value=2.0,value=tries,step=0.1)
        supported_breaks = st.slider(label="Supported Breaks",min_value=-1.0,max_value=1.0,value=supported_breaks,step=0.1)
        defenders_beaten = st.slider(label="Defenders Beaten",min_value=-1.0,max_value=1.0,value=defenders_beaten,step=0.1)
        jackals_success = st.slider(label="Jackals Success",min_value=-1.0,max_value=1.0,value=jackals_success,step=0.1)
        intercepts = st.slider(label="Intercepts",min_value=-1.0,max_value=1.0,value=intercepts,step=0.1)
        tackles_made = st.slider(label="Tackles Made",min_value=-1.0,max_value=1.0,value=tackles_made,step=0.1)
        tackles_missed = st.slider(label="Tackles Missed",min_value=-1.0,max_value=1.0,value=tackles_missed,step=0.1)
        offloads = st.slider(label="Offloads",min_value=-1.0,max_value=1.0,value=offloads,step=0.1)
        carries = st.slider(label="Carries",min_value=-1.0,max_value=1.0,value=carries,step=0.1)
        carry_metres = st.slider(label="Carry Metres",min_value=-1.0,max_value=1.0,value=carry_metres,step=0.1)
        carries_dominant = st.slider(label="Carries Dominant",min_value=-1.0,max_value=1.0,value=carries_dominant,step=0.1)
        turnovers_conceded = st.slider(label="Turnovers Conceded",min_value=-1.0,max_value=1.0,value=turnovers_conceded,step=0.1)
        penalties_conceded = st.slider(label="Penalties Conceded",min_value=-1.0,max_value=1.0,value=penalties_conceded,step=0.1)
        carry_metres_post_contact = st.slider(label="Carry Metres Post Contact",min_value=-1.0,max_value=1.0,value=carry_metres_post_contact,step=0.1)
        ruck_arrivals_attack_first2 = st.slider(label="Ruck Arrivals Attack First2",min_value=-1.0,max_value=1.0,value=ruck_arrivals_attack_first2,step=0.1)
        ruck_arrivals_defence_first2 = st.slider(label="Ruck Arrivals Defence First2",min_value=-1.0,max_value=1.0,value=ruck_arrivals_defence_first2,step=0.1)
        return kick_errors,rucks_won,rucks_lost,linebreaks,tries,supported_breaks,defenders_beaten,jackals_success,intercepts,tackles_made,tackles_missed,offloads,carries,carry_metres,carries_dominant,turnovers_conceded,penalties_conceded,carry_metres_post_contact,ruck_arrivals_attack_first2,ruck_arrivals_defence_first2
def weighting_bar_graph(kick_errors,rucks_won,rucks_lost,linebreaks,tries,supported_breaks,\
                        defenders_beaten,jackals_success,intercepts,tackles_made,tackles_missed,\
                        offloads,carries,carry_metres,carries_dominant,turnovers_conceded,penalties_conceded,\
                        carry_metres_post_contact,ruck_arrivals_attack_first2,ruck_arrivals_defence_first2):
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
    return fig
   