# Machine Learning-Based Flight Anomaly Detection Incorporating Arrival Airport and Runway Context
PROJECT SUMMARY
The primary objective of this proejct was to to develop and compare different machine learning models to detect flight anomalies for commercial aircraft. The second objective was to determine the factors contributing to the risk of flight anomalies. The project used the NASA DASHlink Multi-Class Anomaly Classification Dataset. The dataset was made publicly available through NASA's DASHlink repository. It contains flight trajectories during approach to landing classified as Nominal, Speed High, Path High, and Flaps Late. For this project, the three anomalous classes were combined into a single anomaly class to formulate the problem as binary anomaly detection. The time-series observations for each flight were then aggregated into flight-level features using the mean and standard deviation of the selected flight variables.

Exploratory data analysis showed that anomalous flights accounted for 10% of the total observations, indicating that the resulting binary target variable was imbalanced. Several supervised and unsupervised machine learning algorithms were fitted and compared, including:

Balanced Random Forest (BRF)
Isolation Forest (IF)
Extreme Gradient Boosting Classifier (XGBoost) (with/without SMOTE resampling)
Autoencoder (AutoEnc)
Logistic Regression (LG) (with/without SMOTE resampling)
Artificial Neural Networks (NN)
The training set was used for model development and hyperparameter tuning, while the test set was reserved for final performance evaluation. Model performance was assessed using three widely adopted classification metrics: precision, recall, and F1-score. Given the class imbalance in the dataset, particular emphasis was placed on recall and F1-score, as they provide more informative measures of the model’s ability to identify the minority (anomaly) class than overall classification accuracy.

The comparative analysis demonstrated that XGBoost achieved the highest predictive performance, closely followed by the Neural Network and Balanced Random Forest models, highlighting the effectiveness of ensemble and deep learning approaches for this regression task.

Among the six candidate models, XGBoost demonstrated the strongest overall performance, achieving the highest F1-score (0.790) while maintaining high Precision (0.821) and Recall (0.762). Accordingly, XGBoost was selected as the final model for anomaly detection. However, Balanced Random Forest (BRF) achieved the highest Recall (0.921) and may be preferable in cases where minimizing missed anomalies is prioritized over reducing false-positive detections.

The SHAP analysis identified airspeed_mean as the most influential predictor, followed by tot_pressure_std, altitude_std, and airspeed_std. Higher values of airspeed_mean and altitude_std were generally associated with increased anomaly predictions, whereas higher values of tot_pressure_std and airspeed_std were associated with decreased anomaly predictions. Overall, the results highlight airspeed, total-pressure variability, and altitude variability as key contributors to the XGBoost anomaly predictions.

Project Merit

The primary contribution of this project is that it has incorporated arrival airport and runway into the predictive model as contextual features. This distinguishes it from prior studies that relied primarily on flight-specific characteristics. The inclusion of the two factors above improved prediction performance by about 10%.

Another contribution of this project is the application of SHAP (SHapley Additive exPlanations) to interpret the final XGBoost model and identify the factors that contribute to anomaly predictions. Rather than evaluating the model solely on predictive performance, the SHAP analysis provides insight into the relative importance and direction of influence of individual predictors on the model output. This enables the interpretability of the anomaly-detection framework and provides a better understanding of the flight characteristics and operational factors associated with predicted anomalous behavior.


