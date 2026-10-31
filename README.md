## **PROJECT SUMMARY**

The primary objective of this proejct was to to develop and compare different machine learning models to detect flight anomalies for commercial aircraft.
The second objective was to determine the factors contributing to the risk of flight anomalies. The project used the **NASA DASHlink Multi-Class Anomaly Classification Dataset**. The dataset was made publicly available through NASA's DASHlink repository. It contains flight trajectories during approach to landing classified as **Nominal, Speed High, Path High, and Flaps Late**. For this project, the three anomalous classes were combined into a single anomaly class to formulate the problem as binary anomaly detection. The time-series observations for each flight were then aggregated into flight-level features using the **mean and standard deviation** of the selected flight variables.

Exploratory data analysis showed that anomalous flights accounted for **10%** of the total observations, indicating that the resulting binary target variable was **imbalanced**. Several supervised and unsupervised machine learning algorithms were fitted and compared, including:  

- Balanced Random Forest (**BRF**)
- Isolation Forest (**IF**)
- Extreme Gradient Boosting Classifier (**XGBoost**) *(with/without SMOTE resampling)*
- Autoencoder (**AutoEnc**)
- Logistic Regression (**LG**) *(with/without SMOTE resampling)*
- Artificial Neural Networks (**NN**)  

The training set was used for model development and hyperparameter tuning, while the test set was reserved for final performance evaluation. Model performance was assessed using three widely adopted classification metrics: **precision**, **recall**, and **F1-score**. Given the class imbalance in the dataset, particular emphasis was placed on recall and F1-score, as they provide more informative measures of the model’s ability to identify the minority (anomaly) class than overall classification accuracy.  

The comparative analysis demonstrated that **XGBoost achieved the highest predictive performance**, closely followed by the **Neural Network** and **Balanced Random Forest** models, highlighting the effectiveness of ensemble and deep learning approaches for this regression task.  

Among the six candidate models, **XGBoost demonstrated the strongest overall performance**, achieving the highest F1-score (**0.790**) while maintaining high Precision (**0.821**) and Recall (**0.762**). Accordingly, **XGBoost was selected as the final model** for anomaly detection. However, **Balanced Random Forest (BRF)** achieved the highest Recall (**0.921**) and may be preferable in cases where minimizing missed anomalies is prioritized over reducing false-positive detections.

---

The **SHAP analysis** identified `airspeed_mean` as the most influential predictor, followed by `tot_pressure_std`, `altitude_std`, and `airspeed_std`. Higher values of `airspeed_mean` and `altitude_std` were generally associated with **increased anomaly predictions**, whereas higher values of `tot_pressure_std` and `airspeed_std` were associated with **decreased anomaly predictions**. Overall, the results highlight **airspeed, total-pressure variability, and altitude variability** as key contributors to the XGBoost anomaly predictions.


---

**Project Merit**

The primary contribution of this project is that it has incorporated **arrival airport** and **runway** into the predictive model as contextual features. This distinguishes it from prior studies that relied primarily on flight-specific characteristics. The inclusion of the two factors above improved prediction performance by **about 10%**.

Another contribution of this project is the application of **SHAP (SHapley Additive exPlanations)** to interpret the final XGBoost model and identify the factors that contribute to anomaly predictions. Rather than evaluating the model solely on predictive performance, the SHAP analysis provides insight into the **relative importance and direction of influence of individual predictors** on the model output. This enables the interpretability of the anomaly-detection framework and provides a better understanding of the flight characteristics and operational factors associated with predicted anomalous behavior.


---
---

## **Dataset Description**

### **Target Variable Distribution and Binary Classification**

| Class Index | Original Class | Count | Binary Outcome |
|:-----------:|--------------------|------:|----------------|
| 0 | Nominal | 89,663 | Normal Class |
| 1 | Speed High | 7,013 | Anomaly Class |
| 2 | Path High | 2,207 | Anomaly Class |
| 3 | Flaps Late Setting | 954 | Anomaly Class |
| **Total** | | **99,837** | |

---
---


### **Flight Features Used for Anomaly Detection**
It should be noted that the flap position (`flap_pos`) variable and its derived features (mean and standard deviation) were excluded to prevent potential target leakage, as flap position is directly associated with the definition of the **late-flap anomaly**. An initial analysis showed that **`flap_pos_mean` had the strongest influence on the binary anomaly response**, likely due to its direct relationship with late-flap settings. Therefore, **`flap_pos` and its derived features were excluded from the final model development.**


| Variable | Description |
|:---------|:------------|
| flight_id | Unique identifier for each flight |
| aileron_lh_mean | Mean left aileron position (degrees) |
| aileron_lh_std | Standard deviation of left aileron position (degrees) |
| aileron_rh_mean | Mean right aileron position (degrees) |
| aileron_rh_std | Standard deviation of right aileron position (degrees) |
| aoa_mean | Mean corrected angle of attack (degrees) |
| aoa_std | Standard deviation of corrected angle of attack (degrees) |
| altitude_mean | Mean barometric corrected altitude (feet) |
| altitude_std | Standard deviation of barometric corrected altitude (feet) |
| airspeed_mean | Mean computed airspeed relative to the surrounding air (knots) |
| airspeed_std | Standard deviation of computed airspeed (knots) |
| sel_course_mean | Mean selected course for the aircraft/autopilot (degrees) |
| sel_course_std | Standard deviation of selected course (degrees) |
| drift_angle_mean | Mean angle between aircraft heading and aircraft track (degrees) |
| drift_angle_std | Standard deviation of the drift angle (degrees) |
| elevator_l_mean | Mean left elevator position used to control aircraft pitch (degrees) |
| elevator_l_std | Standard deviation of left elevator position (degrees) |
| gl_slp_dev_mean | Mean deviation from the desired glide slope during approach (%) |
| gl_slp_dev_std | Standard deviation of glide-slope deviation (%) |
| sel_heading_mean | Mean selected heading for the aircraft/autopilot (degrees) |
| sel_heading_std | Standard deviation of selected heading (degrees) |
| loc_dev_mean | Mean localizer deviation from the runway centerline during approach (%) |
| loc_dev_std | Standard deviation of localizer deviation (%) |
| core_speed_mean | Mean engine core speed, indicating engine/throttle operating level (%) |
| core_speed_std | Standard deviation of engine core speed (%) |
| tot_pressure_mean | Mean total pressure (millibar) |
| tot_pressure_std | Standard deviation of total pressure (millibar) |
| pitch_angle_mean | Mean angle between the aircraft longitudinal axis and the horizon (degrees) |
| pitch_angle_std | Standard deviation of aircraft pitch angle (degrees) |
| roll_angle_mean | Mean aircraft roll angle (degrees) |
| roll_angle_std | Standard deviation of aircraft roll angle (degrees) |
| rudder_pos_mean | Mean rudder position (degrees) |
| rudder_pos_std | Standard deviation of rudder position (degrees) |
| true_heading_mean | Mean aircraft true heading measured clockwise from true north (degrees) |
| true_heading_std | Standard deviation of true heading (degrees) |
| vert_accel_mean | Mean aircraft vertical acceleration (G) |
| vert_accel_std | Standard deviation of vertical acceleration (G) |
| wind_speed_mean | Mean wind speed (knots) |
| wind_speed_std | Standard deviation of wind speed (knots) |
| arrival_airport | Arrival airport associated with the flight |
| arrival_runway | Runway used or assigned for the aircraft's arrival |  

