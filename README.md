📡 Human-Micro-Activity-Identification-using-mmWave-sensor

Group No-4

Merapala Yaswanth - 25CS4518

Rupam Dutta - 25ME4410

Gorli Venkata Ramana Murthy - 25CS4510

Soumyadeep Datta - 25CS4508

This repository contains a complete end-to-end Machine Learning pipeline for real-time Human Activity Recognition (HAR) using dual mmWave radar sensors and an ESP32-S3 microcontroller.

By combining the RD-03D (tracking x/y coordinates, velocity, angle) and the DFRobot C4001 (tracking micro-motions, breathing, target energy), this system can accurately classify activities like Walking, Sitting, Standing, Entering, Exiting, and Empty Room (No Person) entirely on the edge without cloud processing. Furthermore, it dynamically detects transient actions like Sit-to-Stand using on-device temporal logic.

🚀 Features

100% Edge AI: Runs a complete Random Forest model locally on the ESP32-S3. No cloud API calls required.

Dual Radar Sensor Fusion: Combines macroscopic movement (RD-03D) with microscopic biological signals (C4001).

Hardware-in-the-Loop (HITL) Evaluation: Natively tests real-time accuracy by reading a 20% validation split directly from an SD card before transitioning to live sensing.

Temporal State Machine: Detects instantaneous transitions (e.g., SIT -> STAND) without retraining the ML model.

Robust Ghost Target Filter: Intelligent logic completely eliminates "ghost" radar reflections from empty walls/desks.

Live Confidence Scoring: Calculates real-time prediction stability and displays accuracy percentages on an OLED screen.

🛠️ Hardware Requirements

Microcontroller: ESP32-S3

Sensor 1: RD-03D mmWave Radar (Connected via UART)

Sensor 2: DFRobot C4001 mmWave Radar (Connected via I2C)

Display: OLED Display (SH110X, I2C)

Storage: MicroSD Card Module (Connected via SPI)

Other: Breadboard, Jumper Wires.

📁 Repository Structure

Dataset/ - (Folder) Raw CSV datasets are stored here, usually managed via Google Drive during training.

datacollection/ - (Arduino Project Folder) Contains the Arduino sketch used to collect the initial raw radar telemetry data into CSV format using an automated 10s/30s state machine.

ML_Model_randomForest/ - (Arduino Project Folder)

ML_Model_randomForest.ino - The main Arduino script that manages the sensors, applies live data smoothing, evaluates SD card ground truth, runs the ML prediction, tracks state transitions, and updates the OLED display.

model_data.h - The generated purely C++ Machine Learning model (to be placed inside the ML_Model_randomForest folder for compiling).

generate_sd_test_data.py - Extracts the exact 20% validation split from the Python environment to evaluate true accuracy on the ESP32.

DATA_PREPROCESSING.ipynb - The ETL pipeline notebook. Generates synthetic empty room data, applies 20Hz Rolling Mean smoothing, labels data, removes noise, and balances classes using SMOTE.

Class Clustering Analysis and Model generation.ipynb - Notebook that trains a highly compact, ESP32-optimized Random Forest model and exports it directly to C++ using micromlgen.

GRAPHS.ipynb - Generates side-by-side comparative visual graphs (KDE, Boxplots, Scatter) to analyze radar telemetry per activity.

🧠 The Machine Learning Pipeline

1. Data Processing & Noise Removal

Radar data is notoriously noisy. The ETL pipeline (DATA_PREPROCESSING.ipynb) takes raw 20Hz sensor data and applies a 5-sample rolling mean (0.25 seconds of data) to smooth out micro-fluctuations. Highly correlated features (> 0.85) are mathematically combined using simple averages to reduce model dimensionality.

2. Model Training

The Class Clustering Analysis and Model generation.ipynb notebook trains a Random Forest Classifier via Scikit-Learn. To fit the strict memory constraints of an ESP32 while maximizing accuracy, the model is finely tuned with the following hyperparameters:

n_estimators = 100 (A strong ensemble consensus)

max_depth = 15 (Allows deep pattern learning without excessive memory consumption)

max_leaf_nodes = 100 (Crucial constraint that caps the absolute maximum size of the C++ if/else logic, compressing the model to ~3.1 MB).

3. C++ Porting

We utilize micromlgen to convert the trained Python Random Forest into plain-text C++ if/else statements (model_data.h). No external ML libraries like TensorFlow Lite are required on the ESP32!

💻 How to Run This Project

Part 1: Training the Model (Google Colab / Jupyter)

Mount your Google Drive or place your raw CSV files in the Dataset folder.

Run all cells in DATA_PREPROCESSING.ipynb to clean the data and balance classes using SMOTE.

Don't forget to Replace the address of the dataset in the code before running. for identification purpose i have replaced the address with this words "ADD THE ADDRESS OF THE DATASET".

Run Class Clustering Analysis and Model generation.ipynb. Move the generated model_data.h file into the ML_Model_randomForest/ Arduino folder.

Run generate_sd_test_data.py to generate test_data_for_esp32.csv. Place this file on the root of your MicroSD card.

Part 2: Flashing the ESP32

Open the Arduino IDE. Install Adafruit GFX and Adafruit SH110X via the Library Manager. Install the C4001 zip library manually.

Open ML_Model_randomForest.ino.

At the top of the file, set bool evaluationModeActive = true; if you want to test the model's accuracy against the SD Card dataset, or false to jump straight to live radar sensing.

Compile and flash to your ESP32-S3.

🔍 Advanced Edge Heuristics

1. The "Ghost Target" Filter

One of the main challenges with mmWave radar is "Ghost Targets" — the radar bouncing off static objects in an empty room, causing the AI to hallucinate a person standing perfectly still. The ESP32 implements a deterministic override:

// If the highly-sensitive C4001 detects 0 targets (no micro-motion/breathing)
// AND the RD-03D detects 0 speed, the room is genuinely empty.
if (c4001_raw_targets == 0.0 && rd_velocity_cm_s == 0.0) {
raw_prediction = "NO_PERSON";
}

2. Temporal State Machine (Transitions)

Instead of wasting memory retraining the model to detect 2-second transient actions (which introduces temporal noise), the ESP32 maintains a historical state tracker. If the AI detects a rapid shift from SITTING to STANDING, the hardware logic mathematically deduces a SIT -> STAND transition and briefly flags it on the UI.

3. Hardware-in-the-Loop (HITL) SD Evaluation

When evaluationModeActive is triggered, the ESP32 reads raw test features from the SD card, runs the C++ inference, and compares the output to the ground-truth string. Once processing is complete, it displays the True Cumulative Accuracy before automatically falling back into Live Sensor Mode.

📜 License

This project is open-source. Feel free to fork, modify, and integrate into your own IoT / Smart Home projects!
