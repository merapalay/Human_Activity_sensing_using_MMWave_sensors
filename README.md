📡 Human-Micro-Activity-Identification-using-mmWave-sensor

Group No-4
Merapala Yaswanth - 25CS4518
Rupam Dutta - 25ME4410
Gorli Venkata Ramana Murthy - 25CS4510
Soumyadeep Datta - 25CS4508

This repository contains a complete end-to-end Machine Learning pipeline for real-time Human Activity Recognition (HAR) using dual mmWave radar sensors and an ESP32-S3 microcontroller.

By combining the RD-03D (tracking x/y coordinates, velocity, angle) and the DFRobot C4001 (tracking micro-motions, breathing, target energy), this system can accurately classify activities like Walking, Sitting, Standing, Entering, Exiting, and Empty Room (No Person) entirely on the edge without cloud processing.

🚀 Features

100% Edge AI: Runs a complete Random Forest model locally on the ESP32-S3. No cloud API calls required.

Dual Radar Sensor(mmWave Sensors) Fusion: Combines macroscopic movement (RD-03D) with microscopic biological signals (C4001).

Robust Ghost Target Filter: Intelligent logic completely eliminates "ghost" radar reflections from empty walls/desks.

Live Confidence Scoring: Calculates real-time prediction stability and displays accuracy percentages on an OLED screen.

Complete ETL Pipeline: Includes Jupyter Notebooks for noise removal (Rolling Mean), SMOTE class balancing, and Correlation Heatmap generation.

🛠️ Hardware Requirements

Microcontroller: ESP32-S3

Sensor 1: RD-03D mmWave Radar (Connected via UART)

Sensor 2: DFRobot C4001 mmWave Radar (Connected via I2C)

Display: OLED Display (SH110X, I2C)

Other: Breadboard, Jumper Wires.

For Pin Configration see datacollection folder's ino file

📁 Repository Structure

Dataset/ - (Folder) Raw CSV datasets are stored here, usually managed via Google Drive during training.

datacollection/ - (Arduino Project Folder) Contains the Arduino sketch used to collect the initial raw radar telemetry data into CSV format.

ML_Model_randomForest/ - (Arduino Project Folder)

ML_Model_randomForest.ino - The Arduino script that manages the sensors, applies live data smoothing, runs the ML prediction, and updates the OLED display.

model_data.h - The generated purely C++ Machine Learning model (to be placed inside the ML_Model_randomForest folder for compiling).

DATA_PREPROCESSING.ipynb - The ETL pipeline notebook. Generates synthetic empty room data, applies 20Hz Rolling Mean smoothing, labels data, removes noise, and balances classes using SMOTE.

Class Clustering Analysis and Model generation.ipynb - Notebook that trains a highly compact, ESP32-optimized Random Forest model and exports it directly to C++ using micromlgen.

GRAPHS.ipynb - Generates side-by-side comparative visual graphs (KDE, Boxplots) to analyze radar telemetry per activity.

DFRobot_C4001-master.zip - Required C4001 sensor library for the Arduino IDE.

README.md - This file.

🧠 The Machine Learning Pipeline

1. Data Processing & Noise Removal

Radar data is notoriously noisy. The ETL pipeline (DATA_PREPROCESSING.ipynb) takes raw 20Hz sensor data and applies a 5-sample rolling mean (0.25 seconds of data) to smooth out micro-fluctuations. Highly correlated features (> 0.85) are mathematically combined using simple averages to reduce model dimensionality.

2. Model Training

The Class Clustering Analysis and Model generation.ipynb notebook trains a Random Forest Classifier via Scikit-Learn. To fit the strict memory constraints of an ESP32 while maximizing accuracy, the model is finely tuned with the following hyperparameters:

n_estimators = 100 (A strong ensemble consensus)

max_depth = 15 (Allows deep pattern learning without excessive memory consumption)

max_leaf_nodes = 100 (Crucial constraint that caps the absolute maximum size of the C++ if/else logic, keeping the model lightweight).

3. C++ Porting

We utilize micromlgen to convert the trained Python Random Forest into plain-text C++ if/else statements (model_data.h). No external ML libraries like TensorFlow Lite are required on the ESP32!

💻 How to Run This Project

Part 1: Training the Model (Google Colab / Jupyter)

Mount your Google Drive or place your raw CSV files (gathered using the datacollection Arduino script) in the Dataset folder.

Don't forget to Replace the address of the dataset in the code before running. for identification purpose i have replaced the address with this words "ADD THE ADDRESS OF THE DATASET".

Run all cells in DATA_PREPROCESSING.ipynb to clean the data, generate empty room baselines, balance classes, and output the final processed dataset.

Run Class Clustering Analysis and Model generation.ipynb. This will read the clean data, train the ML model, and generate a new model_data.h file. Move this file into the ML_Model_randomForest/ folder.

Part 2: Flashing the ESP32

Open the Arduino IDE.

Ensure you have installed the required libraries:

Install Adafruit GFX and Adafruit SH110X via the Library Manager.

Manually install the C4001 library: In the Arduino IDE, go to Sketch > Include Library > Add .ZIP Library... and select the DFRobot_C4001-master.zip file provided in this repository.

Open ML_Model_randomForest.ino from the ML_Model_randomForest/ folder (ensure model_data.h is inside the same folder).

Update the Wi-Fi credentials in the .ino file to allow for initial NTP time synchronization.

Compile and flash to your ESP32-S3.

🔍 The "Ghost Target" Filter (Edge Logic)

One of the main challenges with mmWave radar is "Ghost Targets" — the radar bouncing off static objects (desks, walls) in an empty room, causing the AI to hallucinate a person standing perfectly still.

To solve this, the ESP32 code implements an override filter:

// If the highly-sensitive C4001 detects 0 targets (no micro-motion/breathing)
// AND the RD-03D detects 0 speed, the room is genuinely empty.
if (c4001_raw_targets == 0.0 && rd_velocity_cm_s == 0.0) {
raw_prediction = "NO_PERSON";
}

This forces the AI to output NO_PERSON, completely mitigating static wall reflections while preserving the ability to detect humans sitting perfectly still (via breathing detection).

📜 License

This project is open-source. Feel free to fork, modify, and integrate into your own IoT / Smart Home projects!
