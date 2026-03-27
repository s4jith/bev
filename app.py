import streamlit as st
import matplotlib.pyplot as plt
from visualization import plot_scene

st.set_page_config(page_title="Hackathon Demo: Trajectory Prediction", layout="wide")

st.title("🚗 High-Speed VRU Safety Predictor (6-Second Demo)")
st.markdown("""
Welcome to the interactive highway-scale demo! 
Our Ego Vehicle is driving at high speeds, which means it requires **long-range foresight** to prevent emergencies. 

This model tracks **Vulnerable Road Users (Pedestrians, Cyclists, Motorcycles)** using 2 seconds of history, and predicts 3 distinct potential future paths **6 seconds (12 timesteps)** into the future. 

It evaluates everything within a massive **50-meter radius** to calculate attention weights and prevent collisions before they happen.
""")

st.sidebar.header("Scenario Configuration")

scenario = st.sidebar.selectbox("Select a Demo Scenario", [
    "Clean Straight Path (No interaction)",
    "Approaching Neighbor (Social impact)",
    "Crowded Scene",
    "Custom Coordinate Input"
])

# Base configuration for main pedestrian
main_pedestrian = [(0, 0), (10, 0), (20, 0), (30, 0)]
neighbors = []

if scenario == "Clean Straight Path (No interaction)":
    st.markdown("### Scenario: Clear Highway")
    st.info("The target cyclist is moving straight with no obstacles.")

elif scenario == "Approaching Neighbor (Social impact)":
    st.markdown("### Scenario: Emergency Avoidance")
    st.info("Notice how the target reacts to an object 50 meters ahead.")
    # Neighbor moving much slower ahead
    neighbors = [[(80, 5), (80, 5), (80, 5), (80, 5)]]

elif scenario == "Crowded Scene":
    st.markdown("### Scenario: VRU Crowded Traffic")
    st.info("Multiple pedestrians or cyclists navigating a 100-meter stretch.")
    neighbors = [
        [(80, 5), (82, 5), (84, 5), (86, 5)],   # Slow car ahead
        [(10, -5), (25, -5), (40, -5), (55, -5)],   # Fast overtaking car from behind
        [(50, 10), (55, 10), (60, 10), (65, 10)]    # Car far parallel
    ]

elif scenario == "Custom Coordinate Input":
    st.markdown("### Scenario: Custom Coordinate Input")
    st.info("Provide exactly 4 historical (x,y) points for the pedestrian. The last point is the current position (t=0). Format: x,y separated by semicolons.")
    
    st.markdown("**Example High-Speed Custom Setup:**")
    raw_main = st.text_input("Target VRU Points (t=-3, t=-2, t=-1, t=0)", "0,0; 10,0; 20,0; 30,0")
    raw_neighbors = st.text_area("Neighbor Points (One neighbor per line)", "80,5; 82,5; 84,5; 86,5\n50,10; 55,10; 60,10; 65,10")
    
    try:
        parsed_main = []
        for pt in raw_main.strip().split(';'):
            if pt.strip():
                x, y = map(float, pt.strip().split(','))
                parsed_main.append((x, y))
        
        if len(parsed_main) != 4:
            st.error(f"Your main pedestrian trajectory must have exactly 4 points. You provided {len(parsed_main)}.")
            st.stop()
            
        main_pedestrian = parsed_main
        neighbors = []
        if raw_neighbors.strip():
            for line in raw_neighbors.strip().split('\n'):
                if line.strip():
                    n_pts = []
                    for pt in line.strip().split(';'):
                        if pt.strip():
                            x, y = map(float, pt.strip().split(','))
                            n_pts.append((x, y))
                    if len(n_pts) != 4:
                        st.error(f"Each neighbor must have exactly 4 points. Faulty line has {len(n_pts)}: {line}")
                        st.stop()
                    neighbors.append(n_pts)
    except Exception as e:
        st.error(f"Error parsing coordinates: {e}. Please use format 'x,y; x,y; x,y; x,y'")
        st.stop()

# Tweak target speed optionally in sidebar
if scenario != "Custom Coordinate Input":
    speed_multiplier = st.sidebar.slider("Target Speed Multiplier", min_value=0.5, max_value=2.0, value=1.0)
    # Start at x=0, step by 10m * speed multiplier each timestep (2 Hz means 10m/0.5s = 20m/s = ~72 km/h)
    adjusted_main = [(0, 0), (10*speed_multiplier, 0), (20*speed_multiplier, 0), (30*speed_multiplier, 0)]
else:
    adjusted_main = main_pedestrian


with st.spinner("Running Inference and Generating Scene..."):
    fig = plot_scene(adjusted_main, neighbors)
    st.pyplot(fig)

st.markdown("---")
st.markdown("### Hackathon Team: [Your Name/Team]")
st.markdown("Built using **PyTorch**, **nuScenes**, and **Attention-LSTM**.")
