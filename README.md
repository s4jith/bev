# BEV Trajectory Prediction Transformer

A High-Speed, Socially-Aware Trajectory Prediction system designed for the nuScenes dataset. 
This project goes beyond pedestrian modeling by predicting massive 100-meter highway-speed trajectories **6 seconds into the future**.

## 🚀 Key Technical Features
1. **Transformer Architecture**: Replaced standard LSTM with a `nn.TransformerEncoder` handling Temporal Positional Encodings.
2. **Social Attention**: Uses a MultiHead Attention mechanism computing interaction weights dynamically between vehicles within a **50-meter radius**.
3. **Goal-Conditioned Decoding**: Divides trajectory prediction into two tasks: first predicting the endpoint (Goal), then predicting the physical path to reach it, yielding 3 separate diverse mode probabilities.
4. **HD Map Render Synthesis**: Intercepts native raw image rasters from the `v1.0-mini` metadata to dynamically crop and paint underlay HD Maps for visual collision debugging.

## ⚙️ How to use
**Activate Virtual Environment:**
```bash
.\venv\Scripts\Activate.ps1
```

**1. Train the Core Transformer:**
```bash
python train.py
```
*Current configuration computes ADE, FDE, and diversity pushes over 50 epochs on GPU execution.*

**2. Generate Hackathon Metrics Report:**
```bash
python evaluate.py
```
*Calculates global ADE, FDE, and standard Miss Rate metrics required by judges.*

**3. Run the Interactive Dashboard:**
```bash
streamlit run app.py
```
*Runs the custom prediction engine. Accepts custom coordinate points, evaluates real-time neighbor distances, formats an attention matrix, and plots paths dynamically over the actual dataset's Bird's Eye View map.*

## 🛣️ Project Evolution
* Initially built as a 3-second pedestrian pathfinder.
* Scaled array parameters to intercept real-time vehicle movement up to 100km/h geometry.
* Scaled Future Horizon depth to $t=12$ frames (6.0 seconds) for emergency braking viability.