Customer Churn Prediction

An end-to-end machine learning pipeline that predicts whether a telecom customer is likely to churn, built on the Telco Customer Churn dataset. Four classifiers are trained on a SMOTE-balanced training set and compared on a held-out test set.

Highlights
Dataset: 7,043 customers, 21 raw columns (30 features after encoding)
Class imbalance: about 26.5% churners, handled with SMOTE (training set only)
Models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost
Best model: Random Forest with ROC-AUC 0.819 and accuracy 76.9%
Top churn drivers: TotalCharges, tenure, MonthlyCharges, electronic-check payment, fiber-optic internet, contract type
Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, joblib

Dataset

Telco Customer Churn dataset (one row per customer).

Group	Columns
Demographics	gender, SeniorCitizen, Partner, Dependents
Account	tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Services	PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Target	Churn (Yes / No)
Workflow
Load and inspect: shape, dtypes, null checks.
Clean:
Dropped customerID.
TotalCharges contained blank strings (customers with tenure 0). These were replaced with 0 and the column was cast to float.
Encode:
Target mapped to 0/1.
Categorical features one-hot encoded with pd.get_dummies(drop_first=True), which gives 30 features.
Split: 80/20 stratified train/test split (random_state=42), with 5,634 training rows and 1,409 test rows.
Feature importance: ranked features with a Random Forest.
Balance: applied SMOTE to the training set only, so there is no leakage into the test data.
Train and evaluate: fit four models and compared accuracy, ROC-AUC, precision, recall, and F1.
Results

Evaluated on the 1,409-row test set:

Model	              Accuracy	    ROC-AUC	Churn Precision	Churn Recall	Churn F1
Logistic Regression	0.747	           0.800	  0.52	           0.62	      0.57
Decision Tree	      0.720	           0.658	  0.47	           0.52	      0.50
Random Forest	      0.769	           0.819	  0.56	           0.58	      0.57
XGBoost	            0.770	           0.813	  0.56	           0.59	      0.58

Takeaways

Random Forest and XGBoost perform almost identically. Random Forest has the best ROC-AUC.
Logistic Regression has the highest churn recall, which is useful when catching churners matters most.
The Decision Tree overfits and is the weakest model.
Top 10 Features (Random Forest importance)
Feature	                            Importance
TotalCharges	                        0.191
tenure	                              0.175
MonthlyCharges	                      0.169
PaymentMethod_Electronic check	      0.039
InternetService_Fiber optic	          0.037
Contract_Two year	                    0.030
OnlineSecurity_Yes	                  0.030
gender_Male	                          0.029
PaperlessBilling_Yes	                0.025
TechSupport_Yes	                      0.023
