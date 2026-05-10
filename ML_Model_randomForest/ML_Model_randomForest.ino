/*
 * 1. Generate model_data.h and test_data_for_esp32.csv from Python.
 * 2. Put test_data_for_esp32.csv onto your SD Card.
 * 3. Set EVALUATION_MODE = true to test SD card ground truth.
 * 4. Set EVALUATION_MODE = false to use live radar sensors.
 */

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include "DFRobot_C4001.h"

// Include the purely converted C++ Random Forest model
#include "model_data.h" 

// Instantiate the Random Forest model class
Eloquent::ML::Port::RandomForestModel rf_model;

// ==============================================================
// 🎯 MODE TOGGLE 🎯
// If true, it starts by reading from the SD Card test data. 
// When finished, it will automatically switch to false (Live Mode).
// ==============================================================
bool evaluationModeActive = true; 

// ---------------- SD CARD SETTINGS ----------------
#define SD_CS   10
#define SD_SCK  12
#define SD_MISO 13
#define SD_MOSI 11

File testFile;
int total_eval_samples = 0;
int correct_eval_samples = 0;

// ---------------- PIN SETTINGS ----------------
#define C4001_SDA 6
#define C4001_SCL 7

#define OLED_SDA 8
#define OLED_SCL 9
#define OLED_ADDR 0x3C

#define RD03D_RX_PIN 4
#define RD03D_TX_PIN 5
#define RD03D_BAUD 256000

#define C4001_DETECT_THRESHOLD 200

// ---------------- HARDWARE OBJECTS ----------------
HardwareSerial rdSerial(1);
DFRobot_C4001_I2C c4001;
Adafruit_SH1106G display = Adafruit_SH1106G(128, 64, &Wire1, -1);

// ---------------- GLOBAL SENSOR VARIABLES (For Live Mode) ----------------
float rd_present = 0.0, rd_distance_mm = 0.0, rd_velocity_cm_s = 0.0;
float rd_angle_deg = 0.0, rd_x_mm = 0.0, rd_y_mm = 0.0;
float c4001_present = 0.0, c4001_distance_m = 0.0, c4001_velocity_m_s = 0.0; 
float c4001_energy = 0.0, c4001_raw_targets = 0.0, c4001_raw_distance_m = 0.0;
float c4001_raw_velocity_m_s = 0.0, c4001_raw_energy = 0.0;

unsigned long lastInferenceTime = 0;
const unsigned long INFERENCE_INTERVAL = 500; 

// ---------------- SENSOR SMOOTHING (For Live Mode) ----------------
#define SENSOR_SMOOTHING_WINDOW 5
float sensorHistory[10][SENSOR_SMOOTHING_WINDOW]; 
uint8_t sensorHistIndex = 0;
bool sensorHistFilled = false; 

// ---------------- PREDICTION SMOOTHING (For Live Mode) ----------------
#define SMOOTHING_WINDOW 5
String predictionHistory[SMOOTHING_WINDOW];
uint8_t predIndex = 0;

// ---------------- TRANSITION TRACKER ----------------
String last_stable_activity = "NO_PERSON";
unsigned long transitionDisplayTimer = 0;
String transitionMessage = "";

String getSmoothedPrediction(String new_pred, int &out_confidence) {
    predictionHistory[predIndex] = new_pred;
    predIndex = (predIndex + 1) % SMOOTHING_WINDOW;
    int max_count = 0;
    String best_pred = new_pred;
    for (int i = 0; i < SMOOTHING_WINDOW; i++) {
        if (predictionHistory[i].length() == 0) continue;
        int count = 0;
        for (int j = 0; j < SMOOTHING_WINDOW; j++) {
            if (predictionHistory[i] == predictionHistory[j]) count++;
        }
        if (count > max_count) { max_count = count; best_pred = predictionHistory[i]; }
    }
    out_confidence = (max_count * 100) / SMOOTHING_WINDOW;
    return best_pred;
}

void setup() {
    Serial.begin(115200);
    while (!Serial); 
    
    Serial.println("ESP32-S3 Random Forest Started!");
    Serial.println(evaluationModeActive ? "MODE: Starting with SD Card Eval" : "MODE: Starting with Live Sensors");

    // --- INIT OLED DISPLAY ---
    Wire1.begin(OLED_SDA, OLED_SCL);
    if (!display.begin(OLED_ADDR, true)) {
        Serial.println("OLED failed!");
    } else {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SH110X_WHITE);
        display.setCursor(0, 10);
        display.println(evaluationModeActive ? "INIT SD EVAL MODE..." : "INIT LIVE SENSORS...");
        display.display();
    }

    // --- INIT LIVE SENSORS (Always initialize so they are ready for Live Mode) ---
    rdSerial.begin(RD03D_BAUD, SERIAL_8N1, RD03D_RX_PIN, RD03D_TX_PIN);
    Wire.begin(C4001_SDA, C4001_SCL);
    while (!c4001.begin()) { Serial.println("C4001 missing."); delay(500); }
    c4001.setSensorMode(eSpeedMode);
    c4001.setDetectThres(40, 400, C4001_DETECT_THRESHOLD);
    c4001.setFrettingDetection(eOFF);
    
    for(int i = 0; i < SMOOTHING_WINDOW; i++) predictionHistory[i] = "";
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < SENSOR_SMOOTHING_WINDOW; j++) sensorHistory[i][j] = 0.0;
    }

    if (evaluationModeActive) {
        // --- INIT SD CARD FOR EVALUATION ---
        SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
        if (!SD.begin(SD_CS)) {
            Serial.println("SD Card Mount Failed! Skipping to Live Mode.");
            display.setCursor(0, 30); display.print("SD Failed! -> LIVE"); display.display();
            delay(2000);
            evaluationModeActive = false;
        } else {
            testFile = SD.open("/test_data_for_esp32.csv", FILE_READ);
            if (!testFile) {
                Serial.println("File not found! Skipping to Live Mode.");
                display.setCursor(0, 30); display.print("File missing! -> LIVE"); display.display();
                delay(2000);
                evaluationModeActive = false;
            } else {
                Serial.println("Ready to process SD Card ground truth.");
            }
        }
    }
}

void loop() {
    if (evaluationModeActive) {
        // ==============================================================
        // SD CARD EVALUATION MODE (Hardware-in-the-Loop)
        // ==============================================================
        if (testFile.available()) {
            // Read entire line as a string to completely avoid the parseFloat 1-second timeout bug!
            String line = testFile.readStringUntil('\n');
            line.trim(); // Remove any \r or spaces
            
            if (line.length() > 0) {
                float features[10];
                int startIndex = 0;
                
                // Fast String Tokenization
                for (int i = 0; i < 10; i++) {
                    int commaIndex = line.indexOf(',', startIndex);
                    if (commaIndex == -1) break; 
                    features[i] = line.substring(startIndex, commaIndex).toFloat();
                    startIndex = commaIndex + 1;
                }
                String ground_truth = line.substring(startIndex);
                ground_truth.trim();

                if (ground_truth.length() > 0) {
                    // 1. Run the Prediction using the Edge ML Model
                    int pred_idx = rf_model.predict(features);
                    String prediction = class_activitys[pred_idx];
                    
                    // 2. Calculate True Cumulative Accuracy
                    total_eval_samples++;
                    if (prediction == ground_truth) {
                        correct_eval_samples++;
                    }
                    
                    // Update the screen and serial heavily ONLY every 25 samples so it blazes through!
                    if (total_eval_samples % 25 == 0) {
                        float real_accuracy = ((float)correct_eval_samples / total_eval_samples) * 100.0;
                        
                        Serial.print("Row: "); Serial.print(total_eval_samples);
                        Serial.print(" | True: ["); Serial.print(ground_truth);
                        Serial.print("] \t| Pred: ["); Serial.print(prediction);
                        Serial.print("] \t| True Acc: "); Serial.print(real_accuracy);
                        Serial.println("%");
            
                        display.clearDisplay();
                        display.setCursor(0, 0); display.print("EVALUATING TEST DATA");
                        display.setCursor(0, 16); display.print("Row: "); display.print(total_eval_samples);
                        display.setCursor(0, 30); display.print("AI : "); display.print(prediction);
                        display.setCursor(0, 50); display.print("Acc: "); display.print(real_accuracy, 1); display.print("%");
                        display.display();
                    }
                }
            }
        } 
        
        // Use a separate check here so we catch EOF immediately!
        if (!testFile.available()) {
            Serial.println("\n--- SD CARD EVALUATION COMPLETE ---");
            float final_acc = 0.0;
            if (total_eval_samples > 0) {
                final_acc = ((float)correct_eval_samples / total_eval_samples) * 100.0;
            }
            Serial.print("FINAL TRUE ACCURACY: "); Serial.print(final_acc); Serial.println("%");
            
            display.clearDisplay();
            display.setCursor(0, 10); display.print("EVAL COMPLETE!");
            display.setCursor(0, 30); display.print("Final Acc:");
            display.setCursor(0, 45); display.print(final_acc, 2); display.print("%");
            display.display();
            
            Serial.println(">>> SWITCHING TO LIVE SENSOR MODE IN 5 SECONDS <<<");
            delay(5000); // Show final accuracy for 5 seconds
            
            testFile.close();
            evaluationModeActive = false; // Transition to live mode!
            lastInferenceTime = millis(); // Reset timer for live sensing
        }
        
    } else {
        // ==============================================================
        // LIVE SENSOR INFERENCE MODE
        // ==============================================================
        
        // ---> RD-03D SERIAL PARSING LOGIC <---
        while (rdSerial.available()) {
            uint8_t b = rdSerial.read();
            static uint8_t rdBuf[30];
            static uint8_t rdPos = 0;
            const uint8_t RD_HEADER[4] = {0xAA, 0xFF, 0x03, 0x00};
            const uint8_t RD_FOOTER[2] = {0x55, 0xCC};

            if (rdPos < 4) {
                if (b == RD_HEADER[rdPos]) { rdBuf[rdPos++] = b; } 
                else if (b == RD_HEADER[0]) { rdBuf[0] = b; rdPos = 1; } 
                else { rdPos = 0; }
                continue;
            }
            rdBuf[rdPos++] = b;
            if (rdPos >= 30) {
                if (rdBuf[28] == RD_FOOTER[0] && rdBuf[29] == RD_FOOTER[1]) {
                    int16_t x = ((rdBuf[5] & 0x7F) << 8) | rdBuf[4];
                    if ((rdBuf[5] & 0x80) == 0) x = -x;
                    int16_t y = ((rdBuf[7] & 0x7F) << 8) | rdBuf[6];
                    if ((rdBuf[7] & 0x80) == 0) y = -y;
                    int16_t speed = ((rdBuf[9] & 0x7F) << 8) | rdBuf[8];
                    if ((rdBuf[9] & 0x80) == 0) speed = -speed;
                    uint16_t res = ((uint16_t)rdBuf[11] << 8) | rdBuf[10];

                    rd_present = (x != 0 || y != 0 || res != 0) ? 1.0 : 0.0;
                    rd_x_mm = (float)x; rd_y_mm = (float)y;
                    rd_velocity_cm_s = (float)speed;
                    rd_distance_mm = sqrt((float)x * x + (float)y * y);
                    rd_angle_deg = atan2((float)x, (float)y) * 180.0 / PI;
                }
                rdPos = 0;
            }
        }

        // ---> C4001 I2C READING LOGIC <---
        uint16_t targetCount = c4001.getTargetNumber();
        c4001_raw_targets = (float)targetCount;
        if (targetCount > 0) {
            c4001_present = 1.0; 
            c4001_distance_m = c4001.getTargetRange();
            c4001_velocity_m_s = c4001.getTargetSpeed();
            c4001_energy = (float)c4001.getTargetEnergy();
            c4001_raw_distance_m = c4001_distance_m;
            c4001_raw_velocity_m_s = c4001_velocity_m_s;
            c4001_raw_energy = c4001_energy;
        } else {
            c4001_present = 0.0; c4001_distance_m = 0.0; c4001_velocity_m_s = 0.0; c4001_energy = 0.0;
            c4001_raw_distance_m = 0.0; c4001_raw_velocity_m_s = 0.0; c4001_raw_energy = 0.0;
        }

        unsigned long currentMillis = millis();
        if (currentMillis - lastInferenceTime >= INFERENCE_INTERVAL) {
            lastInferenceTime = currentMillis;

            float instantaneous_data[10] = {
                c4001_distance_m, c4001_present, c4001_raw_distance_m, c4001_raw_targets,
                (c4001_energy + c4001_raw_energy) / 2.0, (c4001_velocity_m_s + c4001_raw_velocity_m_s) / 2.0,
                (rd_angle_deg + rd_x_mm) / 2.0, (rd_distance_mm + rd_y_mm) / 2.0, rd_present, rd_velocity_cm_s
            };

            // Hardware Rolling Mean Smoothing
            for(int i=0; i<10; i++) sensorHistory[i][sensorHistIndex] = instantaneous_data[i];
            sensorHistIndex++;
            if (sensorHistIndex >= SENSOR_SMOOTHING_WINDOW) { sensorHistIndex = 0; sensorHistFilled = true; }

            float smoothed_sensor_data[10];
            for(int i=0; i<10; i++) {
                 float sum = 0.0;
                 int count = sensorHistFilled ? SENSOR_SMOOTHING_WINDOW : sensorHistIndex; 
                 if(count == 0) count = 1; 
                 for(int j=0; j<count; j++){ sum += sensorHistory[i][j]; }
                 smoothed_sensor_data[i] = sum / count;
            }

            int predicted_index = rf_model.predict(smoothed_sensor_data);
            String raw_prediction = String(class_activitys[predicted_index]);
            
            // Deterministic Ghost Target Heuristic Filter
            if (c4001_raw_targets == 0.0 && rd_velocity_cm_s == 0.0) raw_prediction = "NO_PERSON";
            
            int live_confidence = 0;
            String predicted_action = getSmoothedPrediction(raw_prediction, live_confidence);
            String display_action = predicted_action; // Default to the standard AI output
            
            // ==============================================================
            // 🔄 STATE MACHINE: TRANSITION DETECTION LOGIC 🔄
            // ==============================================================
            // If the AI suddenly jumps from Sitting to Standing, trigger a transition message!
            if (last_stable_activity == "SITTING" && predicted_action == "STANDING") {
                transitionMessage = "SIT -> STAND";
                transitionDisplayTimer = millis(); // Show this message for 3 seconds
            } 
            else if (last_stable_activity == "STANDING" && predicted_action == "SITTING") {
                transitionMessage = "STAND -> SIT";
                transitionDisplayTimer = millis(); // Show this message for 3 seconds
            }

            // Only update the last_stable_activity if we are fully settled into a state
            // (We ignore Walking or Entering so it doesn't mess up the Sit/Stand logic)
            if (predicted_action == "SITTING" || predicted_action == "STANDING") {
                last_stable_activity = predicted_action;
            }

            // Override the display action if a transition recently occurred (within last 3000ms)
            if (millis() - transitionDisplayTimer < 3000) {
                display_action = transitionMessage;
            }
            // ==============================================================
            
            Serial.print("🌲 Raw: "); Serial.print(raw_prediction);
            Serial.print("  |  🎯 Output: "); Serial.print(display_action);
            Serial.print("  |  Stability: "); Serial.print(live_confidence); Serial.println("%");

            display.clearDisplay();
            display.setCursor(0, 0); display.print(display_action);
            display.setCursor(80, 0); display.print("Stb:"); display.print(live_confidence); display.print("%");
            
            display.setCursor(0, 14); display.print("RV:"); display.print(rd_velocity_cm_s, 0); 
            display.setCursor(64, 14); display.print("C4D:"); display.print(c4001_distance_m, 2);
            display.setCursor(0, 24); display.print("RD:"); display.print(rd_distance_mm, 0);
            display.setCursor(64, 24); display.print("C4V:"); display.print(c4001_velocity_m_s, 2);
            display.setCursor(0, 34); display.print("RA:"); display.print(rd_angle_deg, 0);
            display.setCursor(64, 34); display.print("Tgt:"); display.print(c4001_raw_targets, 0);
            display.setCursor(0, 44); display.print("RX:"); display.print(rd_x_mm, 0);
            display.setCursor(0, 54); display.print("RY:"); display.print(rd_y_mm, 0);
            
            display.display();
        }
    }
}