# Enhancing-Explosive-Motions-in-Humanoid-Robotics

## Enhancing Explosive Motion in Humanoid Robotics with Imitation and Reinforcement Learning
This repository contains the code and resources for the Master Thesis titled "Enhancing Explosive Motion in Humanoid Robotics with Imitation and Reinforcement Learning", submitted to the Institute of Control Theory and Systems Engineering, Faculty of Electrical Engineering and Information Technology, Technische Universität Dortmund. The work was conducted by Shubhankar Kulkarni from Maharashtra, India, with a submission date of September 01, 2025. The responsible professor is Univ.-Prof. Dr.-Ing. Prof. h.c. Dr. h.c. Torsten Bertram, with academic supervisors Apl. Prof. Dr. Frank Hoffmann and M.Sc. Martin Krüger.

## Overview
This project explores the integration of imitation learning and reinforcement learning to enhance explosive motion capabilities in humanoid robotics. The implementation leverages the MuJoCo physics engine and the Proximal Policy Optimization (PPO) algorithm to train a humanoid robot to perform dynamic jumps, with adaptive parameter tuning and motion trail visualization.

## Installation
```bash
git clone https://github.com/yourusername/humanoid-robotics-motion.git
cd humanoid-robotics-motion
```

### Install dependencies:
check ```environment.yml```

### Set up MuJoCo:
Refer: https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da

### Usage
The main script main_mod.py serves as the entry point for training, evaluating, or capturing motion trails of the humanoid robot. Use the following commands based on your needs:

### Train the Model:
Run the training loop with the default configuration:
```bash
python main.py
```

This will train the PPO agent for 30 million steps with the specified hyperparameters (e.g., learning rates, horizon, etc.) and save checkpoints every 500,000 steps.

### Evaluate the Model:
Load a pre-trained model and evaluate its performance by changing the parameter ```humanoidJump.yaml``` file ```render``` to ```True```:
```bash
python main.py
```

Use ```best``` to load the best model instead of a specific index.

## Working
https://github.com/user-attachments/assets/753a337b-c837-46e5-83a7-eb297563c143

## Configuration

The configuration is loaded from ```humanoidJump.yaml```. Modify parameters such as max_train_steps, a_lr, c_lr, or log_std to adjust the training process.
Ensure the model and mocap_file paths in the YAML file point to the correct locations.

Find the full report at [Report](masters_thesis_report.pdf)