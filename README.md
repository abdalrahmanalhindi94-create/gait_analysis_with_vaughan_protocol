# Gait Analysis with Vaughan Protocol

This project implements the **Vaughan kinematic gait analysis protocol** using experimental 3D motion-capture data collected in the university biomechanics laboratory.  
The MATLAB script preprocesses marker trajectories, extracts a full gait cycle, computes anatomical joint angles (hip, knee, ankle), and generates plots for all three planes of movement.

---


---

## 🧪 Dataset Description

The file **HH_experiments.mat** contains:

- `Gait1_kinematics` — 3D marker trajectories from pelvis, thigh, calf, and foot  
- `Gait1_plate_LRF` — force-plate / reference data used to identify gait events  

These signals were collected during a gait trial performed in the biomechanics laboratory.

---

## 🔍 What the MATLAB Script Does

### ✔ 1. Load & preprocess data  
- Crops the raw signals  
- Fixes missing samples using spline interpolation  
- Applies a 5 Hz Butterworth low-pass filter  
- Extracts one full gait cycle from heel-marker data

### ✔ 2. Build anatomical reference frames  
Creates anatomical coordinate systems for:  
- Pelvis  
- Thigh (right & left)  
- Calf (right & left)  
- Foot (right & left)

### ✔ 3. Compute joint angles (Grood & Suntay convention)  
For the **right lower limb**:  
- Hip  
- Knee  
- Ankle  

Each joint includes:  
- **Flexion–Extension**  
- **Abduction–Adduction**  
- **Internal–External Rotation**

### ✔ 4. Plot the results  
The script generates clean plots of each angle over **0–100% of the gait cycle**, including:  
- Initial Contact (IC)  
- Toe Off (TO)  
- Opposite Toe Off (OT)  
- Heel Rise (HR)  
- Opposite Initial Contact (OI)  
- Foot Adjacent (FA)  
- Tibial Vertical (TV)

---

## ▶️ How to Run

1. Open MATLAB  
2. Add project folders to the path:
   ```matlab
   addpath('code')
   addpath('data')
3. Run the main script:

```matlab
vaughan_analysis


