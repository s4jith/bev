import streamlit as st
import matplotlib.pyplot as plt
from visualization import plot_scene

st.set_page_config(page_title="Hackathon Demo: Trajectory Prediction", layout="wide")

st.title("High-Speed VRU Safety Predictor (6-Second Demo)")
st.markdown("""
Welcome to the interactive highway-scale demo!
Our Ego Vehicle is driving at high speeds, which means it requires **long-range foresight** to prevent emergencies.

This model tracks **Vulnerable Road Users (Pedestrians, Cyclists, Motorcycles)** using 2 seconds of history, and predicts 3 distinct potential future paths **6 seconds (12 timesteps)** into the future.

It evaluates everything within a massive **50-meter radius** to calculate attention weights and prevent collisions before they happen.
""")

st.sidebar.header("Scenario Configuration")

scenario = st.sidebar.selectbox("Select a Demo Scenario", [
    "Custom Coordinate Input",
    "Clean Straight Path (No interaction)",
    "Approaching Neighbor (Social impact)",
    "Crowded Scene",
    "Static Obstacle Avoidance (Tree/Standing Person)",
    "DENSE CITY TRAFFIC (Bangalore Simulation)"
])

# Base configuration for main pedestrian
main_pedestrian = [(0, 10), (2, 10), (4, 10), (6, 10)]
neighbors = []
neighbor_types = []

if scenario == "Custom Coordinate Input":
    st.markdown("### Scenario: User Custom Input")
    st.info("Provide exactly 4 historical (x,y) points for the Target VRU.")
    col1, col2 = st.columns(2)
    with col1:
        raw_main = st.text_input("Target VRU Points (t=-3, t=-2, t=-1, t=0)", "0,10; 2,10; 4,10; 6,10")
    with col2:
        raw_neighbors = st.text_area("Neighbor Points (One per line)", "0,12; 0,12; 0,12; 0,12\n10,8; 10,8; 10,8; 10,8")

    try:
        parsed_main = [(float(pt.split(',')[0]), float(pt.split(',')[1])) for pt in raw_main.strip().split(';') if pt.strip()]
        if len(parsed_main) != 4: st.stop()
        main_pedestrian = parsed_main
        neighbors = []
        neighbor_types = []
        if raw_neighbors.strip():
            for line in raw_neighbors.strip().split('\n'):
                if line.strip():
                    n_pts = [(float(pt.split(',')[0]), float(pt.split(',')[1])) for pt in line.strip().split(';') if pt.strip()]
                    if len(n_pts) != 4: st.stop()
                    neighbors.append(n_pts)
                    neighbor_types.append('Car')
    except Exception as e:
        st.stop()

elif scenario == "Clean Straight Path (No interaction)":
    st.markdown("### Scenario: Clear Highway")
    st.info("The target cyclist is moving straight with no obstacles.")

elif scenario == "Approaching Neighbor (Social impact)":
    st.markdown("### Scenario: Emergency Avoidance")
    st.info("Notice how the target reacts to an object 50 meters ahead.")
    neighbors = [[(80, 5), (80, 5), (80, 5), (80, 5)]]
    neighbor_types = ['Car']

elif scenario == "Crowded Scene":
    st.markdown("### Scenario: VRU Crowded Traffic")
    neighbors = [
        [(80, 5), (82, 5), (84, 5), (86, 5)],   
        [(10, -5), (25, -5), (40, -5), (55, -5)],   
        [(6, 8), (7, 8), (8, 8), (9, 8)]    
    ]
    neighbor_types = ['Car', 'Car', 'Person']
    
elif scenario == "Static Obstacle Avoidance (Tree/Standing Person)":
    st.markdown("### Scenario: Avoiding an unmoving Object")
    main_pedestrian = [(0, 10), (1.5, 10), (3.0, 10), (4.5, 10)]
    neighbors = [[(12, 10), (12, 10), (12, 10), (12, 10)]]
    neighbor_types = ['Static/Tree']

elif scenario == "DENSE CITY TRAFFIC (Bangalore Simulation)":
    st.markdown("### Scenario: Extreme Chaos (Bangalore Traffic Multi-Agent Math)")
    st.error("WARNING: Very dense environment. Testing massive parallel Multi-Head Attention scaling.")
    
    # Target pedestrian jaywalking diagonally across the street
    main_pedestrian = [(0, -5), (1, -3), (2, -1), (3, 1)]
    
    neighbors = [
        # Incoming fast cars in main lanes
        [(15, 6), (12, 6), (9, 6), (6, 6)],
        [(30, 2), (25, 2), (20, 2), (15, 2)],
        [(-20, -2), (-15, -2), (-10, -2), (-5, -2)],
        
        # Swerving Auto-Rickshaw / Bikers squeezing between lanes
        [(0, -8), (2, -6), (4, -4), (5, -2)],
        [(10, 15), (11, 12), (12, 9), (13, 6)],
        
        # Pedestrian crowd huddled on the top sidewalk
        [(8, 12), (8, 12), (8, 12), (8, 12)], # Standing person
        [(9, 12), (9, 12), (9, 12), (9, 12)], # Standing person
        [(10, 12), (10, 12), (10, 12), (10, 12)], # Standing person
        
        # Random dogs / static objects in the road
        [(4, 4), (4, 4), (4, 4), (4, 4)], # Trash can or dog
        [(-2, 3), (-2, 3), (-2, 3), (-2, 3)]
    ]
    neighbor_types = [
        'Car', 'Car', 'Car', 
        'Bike', 'Bike', 
        'Person', 'Person', 'Person',
        'Static/Tree', 'Static/Tree'
    ]


if scenario not in ["Custom Coordinate Input", "Static Obstacle Avoidance (Tree/Standing Person)", "DENSE CITY TRAFFIC (Bangalore Simulation)"]:
    speed_multiplier = st.sidebar.slider("Target Speed Multiplier", min_value=0.5, max_value=2.0, value=1.0)
    adjusted_main = [(0, 10), (2*speed_multiplier, 10), (4*speed_multiplier, 10), (6*speed_multiplier, 10)]
else:
    adjusted_main = main_pedestrian


with st.spinner("Running Inference and Generating Scene..."):
    fig = plot_scene(adjusted_main, neighbors, neighbor_types)
    st.pyplot(fig)
