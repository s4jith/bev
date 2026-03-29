# BEV VRU Trajectory Prediction - Complete PPT Content (Detailed)

This document is prepared as direct PPT content for your hackathon presentation.
It is fully aligned with the current project implementation and measured results.

Use each slide section as copy-paste material.

---

## Slide 1 - Title Slide
Title:
BEV VRU Trajectory Prediction Transformer

Subtitle:
High-Speed, Socially-Aware 6-Second Forecasting for Vulnerable Road Users

Presenter line:
Hackathon Final Submission - nuScenes v1.0-mini Based System

Speaker notes:
This project predicts future motion of vulnerable road users such as pedestrians, bicycles, and motorcycles. It focuses on long-horizon safety forecasting to support early and safer autonomous driving decisions.

---

## Slide 2 - What Type of Model Is This?
Title:
Model Category

Slide bullets:
- This is a Trajectory Forecasting model (sequence-to-sequence regression).
- It is not a classification model.
- It is not an object detection model.
- It is not an image segmentation model.
- It predicts future 2D coordinates over time, based on observed motion and social interactions.

Speaker notes:
If someone asks which category this belongs to among classification, segmentation, detection, and tracking/tracing, the best answer is trajectory forecasting under motion prediction. It is closest to tracking/tracing in intent, but technically it is future trajectory regression.

---

## Slide 3 - Key Objective of the Challenge
Title:
Challenge Objective

Slide bullets:
- Predict where VRUs will move 6 seconds into the future.
- Use only short observation history (4 points, around 2 seconds).
- Generate multi-modal future paths (3 possible futures).
- Include social context of neighboring road users within 50 meters.
- Reduce collision risk by giving ego vehicle more reaction time.

Speaker notes:
The objective is not just to predict one path, but to predict plausible alternatives while considering social interactions. This helps planning systems brake or re-plan earlier.

---

## Slide 4 - Real World Impact
Title:
Why It Matters in Real Traffic

Slide bullets:
- Early risk anticipation for pedestrians, cyclists, and motorcycles.
- Better emergency response window for ADAS or autonomous stack.
- Safer behavior in crowded and mixed-traffic scenes.
- Supports highway-speed safety by extending future look-ahead horizon.
- Lightweight model design supports practical real-time deployment.

Speaker notes:
The model improves proactive safety. Instead of reacting late, the vehicle can make earlier and smoother decisions.

---

## Slide 5 - System Overview (End-to-End)
Title:
End-to-End System Flow

Slide bullets:
- Input: nuScenes annotation trajectories and neighbor trajectories.
- Preprocessing: VRU filtering, trajectory linking, sliding windows, feature engineering.
- Inference: Transformer temporal encoding + social attention + goal-conditioned decoding.
- Post-processing: multi-path output, confidence scores, best-path selection for metrics.
- Output: 3 future trajectories, probability per trajectory, BEV visualization overlay.

Speaker notes:
This system is the prediction stage in a larger autonomous stack. It uses kinematic tracks rather than raw camera pixels.

---

## Slide 6 - Typical Computer Vision Pipeline Mapping
Title:
Pipeline Mapping to Standard CV Stages

Slide bullets:
- Input Data
- Preprocessing
- Feature Extraction
- Model Inference
- Post Processing
- Output Prediction

Detailed mapping:
- Input Data: nuScenes v1.0-mini metadata and annotation trajectories.
- Preprocessing: filter VRUs, construct temporal windows, center coordinates, build neighbor sets.
- Feature Extraction: per timestep features [x, y, dx, dy, speed, sin(theta), cos(theta)].
- Model Inference: transformer encoder, social attention pooling, goal-conditioned multimodal decoder.
- Post Processing: trajectory mode probabilities and metric-oriented best-of-K selection.
- Output Prediction: 12-step future trajectories for 6 seconds, with confidence.

Speaker notes:
Even though this is not pixel-heavy CV in this stage, the processing still follows the standard AI pipeline structure.

---

## Slide 7 - Block Diagram of System Architecture
Title:
Architecture Block Diagram

Use this diagram directly on slide:

```text
[nuScenes v1.0-mini JSON annotations]
            |
            v
[VRU Filtering: pedestrian / bicycle / motorcycle]
            |
            v
[Trajectory Linking Over Time]
            |
            v
[Sliding Windows: 4 observed + 12 future]
            |
            v
[Kinematic Features per timestep:
 x, y, dx, dy, speed, sin(theta), cos(theta)]
            |
            +------------------------------+
            |                              |
            v                              v
[Target Sequence Encoder]          [Neighbor Sequences]
(Transformer Encoder)              (within 50m)
            |                              |
            +--------------+---------------+
                           v
                 [Social Multihead Attention]
                           |
                           v
                 [Context Fusion (target + social)]
                           |
            +--------------+----------------+
            |                               |
            v                               v
      [Goal Head: K=3 endpoints]     [Prob Head: mode scores]
            |
            v
 [Goal-Conditioned Trajectory Head]
            |
            v
[3 Future Trajectories x 12 steps + probabilities]
            |
            v
[Evaluation Metrics + BEV Visualization]
```

Speaker notes:
The key novelty is combining temporal encoding, social interaction attention, and goal-conditioned decoding to produce realistic multi-modal futures.

---

## Slide 8 - Step-by-Step Workflow
Title:
Step-by-Step Workflow

Training workflow:
1. Load nuScenes annotation tables from v1.0-mini.
2. Filter VRU categories (pedestrian, bicycle, motorcycle).
3. Build full instance trajectories using linked annotations.
4. Create windows of 16 timesteps (4 observed + 12 future).
5. Compute kinematic features and neighbor context in 50 m radius.
6. Train Transformer model with multimodal objective.
7. Save best checkpoint based on validation ADE trend.

Inference workflow:
1. Receive 4 recent points for target VRU.
2. Convert points to kinematic feature tensor.
3. Gather optional neighbor histories.
4. Run model forward pass.
5. Get 3 predicted trajectories and probabilities.
6. Render trajectories on BEV map patch.

Speaker notes:
This clear step chain is useful when judges ask where each module starts and ends.

---

## Slide 9 - Core Model Architecture (Implementation-Level)
Title:
Inside the Model

Slide bullets:
- Input shape: (B, 4, 7)
- Embedding: Linear 7 -> 64
- Positional encoding: sinusoidal temporal encoding
- Transformer encoder: 2 layers, 4 heads, feed-forward 256
- Social interaction: MultiheadAttention with 4 heads
- Multi-modal outputs: K = 3 future modes
- Future horizon: 12 points (6 seconds)

Decoder heads:
- Goal head: predicts 3 endpoints (x, y)
- Trajectory head: conditioned on context + goal
- Probability head: softmax confidence for each mode

Speaker notes:
The model separates where the target will end up and how it gets there. This improves long-horizon trajectory realism.

---

## Slide 10 - Core Algorithms Used
Title:
Algorithms and Techniques

Algorithm usage summary:
- Transformer sequence modeling: Used
- Positional encoding: Used
- Social multihead attention: Used
- Goal-conditioned multimodal decoding: Used
- Best-of-K trajectory selection in loss and eval: Used
- Diversity regularization across modes: Used

Requested examples mapping:
- CNN architectures (ResNet / EfficientNet): Not used in this stage
- YOLO / UNet detection-segmentation: Not used in this stage
- Feature extraction method: Kinematic engineered features from trajectory coordinates
- Transfer learning strategy: Not currently used (trained from scratch)
- Optimization methods: Adam optimizer + custom composite loss

Speaker notes:
This project targets the prediction layer and assumes upstream perception/tracking has already provided agent coordinates.

---

## Slide 11 - Data Training Strategy: Dataset
Title:
Dataset and Data Source

Slide bullets:
- Dataset: nuScenes (public), split used: v1.0-mini
- Input source files: category.json, instance.json, sample_annotation.json
- Filtered target classes: pedestrian, bicycle, motorcycle
- Built trajectories: 195
- VRU instances considered: 264
- Generated training/evaluation samples: 2,338

Speaker notes:
This is a public benchmark dataset. We transformed raw annotation streams into fixed temporal windows for sequence learning.

---

## Slide 12 - Data Training Strategy: Preprocessing
Title:
Preprocessing and Feature Engineering

Slide bullets:
- Build continuous trajectory chains from prev-next annotation links.
- Use sliding windows over each trajectory.
- Window structure: 4 observed points + 12 future points.
- Coordinate normalization: center around last observed point.
- Derive kinematic features at each timestep.
- Build social neighborhood set inside 50 m radius.

Kinematic features per timestep:
- x, y
- dx, dy
- speed = sqrt(dx^2 + dy^2)
- sin(theta), cos(theta)

Speaker notes:
Feature engineering allows efficient, low-latency learning without heavy pixel processing.

---

## Slide 13 - Data Strategy: Augmentation, Splits, and Hardware
Title:
Training Protocol

Slide bullets:
- Data augmentation: No explicit augmentation currently implemented.
- Split strategy: 80% train, 20% validation.
- Batch size: 64
- Epochs: 50
- Optimizer: Adam, learning rate 0.001
- Hardware: CUDA GPU used when available

Important note:
- Random split is currently done without fixed seed.
- Therefore, exact train/validation partition can vary run-to-run.

Speaker notes:
This is a strong baseline setup. For stricter reproducibility, set deterministic random seeds in future iterations.

---

## Slide 14 - Loss Design and Optimization
Title:
Training Objective

Composite loss components:
- Trajectory loss: Best-of-K path error against ground truth.
- Goal loss: endpoint supervision to improve final destination accuracy.
- Probability loss: mode classification toward best path index.
- Diversity loss: penalizes mode collapse across predicted trajectories.

Loss form used in code:
Total Loss = TrajectoryLoss + 0.5 * GoalLoss + 0.5 * ProbLoss + 0.1 * DiversityLoss

Why this helps:
- Best-of-K supports multi-modal uncertainty.
- Goal conditioning improves long-horizon final-point consistency.
- Diversity term avoids nearly identical future hypotheses.

Speaker notes:
This objective balances accuracy and diversity, which is crucial for realistic future forecasting.

---

## Slide 15 - Real-Time Processing and Performance
Title:
Performance Effectiveness

Model efficiency:
- Parameter count: 146,017
- Checkpoint size: 625,598 bytes (about 0.63 MB)

Measured inference speed (GPU, forward-pass benchmark):
- No neighbors:
  - Average latency: 1.143 ms
  - P95 latency: 1.436 ms
  - Throughput: about 874.72 FPS
- With 10 neighbors:
  - Average latency: 7.969 ms
  - P95 latency: 8.970 ms
  - Throughput: about 125.49 FPS

Latency handling points:
- Kinematic input avoids expensive image backbone at this stage.
- Social attention increases cost with neighbor count.
- Still real-time under tested social-load scenarios.

Speaker notes:
The model is compact and fast enough for real-time planning loops, especially with bounded neighbor counts.

---

## Slide 16 - Edge Deployment Capability
Title:
Deployment Readiness

Slide bullets:
- Small model footprint supports edge potential.
- Fast GPU inference supports high-frequency updates.
- Suitable for integration into AV prediction modules.
- Can be exported to optimized runtimes in future (TorchScript/ONNX).

Operational recommendations:
- Set fixed cap on maximum neighbors to bound latency.
- Add quantization and runtime profiling for embedded hardware.
- Use deterministic preprocessing pipeline for deployment consistency.

Speaker notes:
This architecture is deployment-friendly compared to large pixel-heavy models.

---

## Slide 17 - Robustness Analysis
Title:
Robustness: Noise, Lighting, Occlusion

Noise robustness:
- Moderate robustness to coordinate-level noise.
- Performance depends on quality of upstream tracking trajectories.

Lighting variation:
- Strongly robust in this prediction stage because model uses kinematics, not raw image pixels.

Occlusions:
- If upstream tracker loses continuity under occlusion, trajectory history quality drops.
- Prediction quality then may degrade because this model depends on trajectory continuity.

Speaker notes:
The key robustness dependency is upstream perception and tracking reliability.

---

## Slide 18 - Evaluation Metrics
Title:
How We Measure Performance

Primary forecasting metrics used:
- ADE (Average Displacement Error): mean distance error across all future timesteps.
- FDE (Final Displacement Error): endpoint error at final timestep.
- Miss Rate (>2.0 m): percentage of trajectories whose final point is more than 2 meters away.

Formulas:
- ADE = average over all samples and timesteps of L2(predicted_t - groundtruth_t)
- FDE = average over all samples of L2(predicted_final - groundtruth_final)
- MissRate = (count of samples with final error > 2.0 m) / total samples

Requested CV metrics note:
- Accuracy / Precision / Recall / F1 / mAP / IoU are standard for classification or detection/segmentation.
- They are not primary KPIs for trajectory forecasting.
- For this project, ADE/FDE/Miss Rate are the correct task metrics.

Speaker notes:
If judges ask for accuracy, use trajectory-success proxy based on miss threshold.

---

## Slide 19 - Final Results and Accuracy Interpretation
Title:
Final Quantitative Results

Measured on evaluation script:
- Total trajectories evaluated: 2,338
- Prediction horizon: 6 seconds (12 steps)
- Social context radius: 50 meters
- ADE: 0.6967 m
- FDE: 1.3306 m
- Miss Rate (>2.0 m): 7.96%

Accuracy-style interpretation:
- Success rate within 2.0 m final threshold:
- 100% - 7.96% = 92.04%

Speaker notes:
For trajectory forecasting, this threshold-based success value is a practical way to communicate accuracy to non-technical audiences.

---

## Slide 20 - Benchmark Comparison
Title:
Benchmarking Status

Available comparisons now:
- Internal latency comparison with and without social neighbors.
- Training trend monitoring across epochs.
- Final evaluation metrics on full prepared dataset.

Not currently available:
- External baseline comparison against other models in this repository.
- mAP/IoU style benchmark tables (not task-aligned for this model type).

Suggested benchmark roadmap:
- Add baseline models: constant velocity, LSTM baseline, non-social transformer variant.
- Compare ADE/FDE/Miss Rate under identical split and seed.
- Report latency-vs-accuracy tradeoff.

Speaker notes:
Be transparent: we have strong internal results, and external baseline comparisons are a clear next step.

---

## Slide 21 - Tech Stack Summary
Title:
Technology Stack

Core:
- Python
- PyTorch (CUDA)
- NumPy
- Matplotlib
- Streamlit

Data and tooling:
- nuScenes devkit
- pandas
- OpenCV-headless
- shapely
- pyquaternion

System modules in this project:
- Data processing and sample generation
- Transformer model definition
- Training and validation pipeline
- Evaluation metrics report
- Inference API
- BEV map rendering and dashboard demo

Speaker notes:
The stack is practical, reproducible, and suitable for both research demo and iterative productization.

---

## Slide 22 - Limitations and Next Steps
Title:
Current Limitations and Future Work

Current limitations:
- No explicit data augmentation.
- No fixed random seed for deterministic split.
- No direct external model benchmark in current repo.
- Depends on upstream tracking quality under occlusion-heavy scenes.

Next improvements:
- Add deterministic split and reproducibility controls.
- Add baseline model comparison table.
- Add uncertainty calibration and confidence quality analysis.
- Optimize for edge deployment with quantization.
- Expand to larger nuScenes splits for stronger generalization.

Speaker notes:
This closes the presentation with realism and a clear technical roadmap.

---

## Appendix - One-Slide Quick Facts
Title:
Quick Facts for Judges

- Model type: Trajectory forecasting (sequence regression)
- Core network: Transformer + social multihead attention + goal-conditioned decoder
- Dataset: nuScenes v1.0-mini (public)
- Samples used: 2,338
- Future horizon: 6 seconds (12 points)
- Modes predicted: 3
- ADE: 0.6967 m
- FDE: 1.3306 m
- Miss Rate (>2.0 m): 7.96%
- Approx success within 2.0 m: 92.04%
- Model size: about 0.63 MB
- Parameters: 146K

Use this as a backup summary slide or final closing slide.
