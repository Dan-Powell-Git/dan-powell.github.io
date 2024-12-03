# Portfolio


## Home Credit Default Prediction 

<img src ="assets/img/Home-Credit-logo.jpg" width = "20%" > 

### Summary of Business Problem: 
Identify customers with a high risk of default on their loans to improve repayment rates at HomeCredit while allowing for disadvantaged groups to receive loans to jumpstart
their businesses or personal projects. 

### Objective: 
Predict loan defaults using machine learning methods on the Home Credit Kaggle Dataset

### Data: 
-	Application_train: contains customer information and the target variable (whether the customer defaulted)
-	Application_test: contains customer information without the target variable used for testing the performance of models

### EDA: 
My EDA cleaned, transformed, and informed the direction of the project. The EDA highlighted several challenges with the data including:
-	Class imbalance in the target variable
-	Extreme observations in several of the independent variables
-	Redundant and noninformative independent variables
To address these limitations:
-	I performed exploratory analysis to understand the distribution of the target and independent variables, missing values, and outliers
-	I engineered new features based off correlation analyses, domain knowledge, and data insights
The dataset was refined through null imputation, class balancing with SMOTE, and the incorporation of new features inspired by insights from the EDA.

### Modeling:
I constructed 3 models as a part of a group project. I was assigned to train SVM methods on the data. 
From this, constructed three models:
-	**Linear SVM**: basic linear SVM model with no adjustments to training data or hyperparameters
-	**Linear SVM with SMOTE**: Linear SVM trained on SMOTE-balanced data with an adjusted classification threshold.
-	**Radial SVM with SMOTE and Adjustments**: Radial SVM trained on SMOTE data with higher weights on default observations and an optimized classification threshold using a function that optimized the F1 score produced by the threshold.

One challenge during this stage was training the radial and linear SVM with SMOTE models. Due to the large size of the training data, the training time for the two previously mentioned models 
lasted over 3 hours before eventually crashing. To address this, I implemented a data sampling strategy. By randomly selecting a subset of 5000 observations, I was able to maintain model generalizability
while significantly improving training efficiency. This approach allowed for the successful training of more sophisticated models that would otherwise have been infeasible.

After the models were ran, I observed that the radial SVM with SMOTE-balanced data performed best in the test split (F1: 0.175, accuracy: 0.64) and achieved a score of 0.58 on Kaggle. 
The model was then compared to the models my group members produced, which were:
-	**LightGBM**: Utilized a gradient boosting machine for predictions
-	**Logistic Regression**: employed logistic regression for simpler and interpretable model

### Performance Metrics of models:
Each model was evaluated on the test data from Kaggle. Kaggle scores reflect the model's ability to classify the test data correctly and how it ranks compared to other models.
-	SVM: Kaggle of was 0.58.
-	Logistic Regression: Kaggle Score of 0.73
-	LightGBM: Achieved Kaggle Score of 0.74

### Key Findings and Solutions:
From the modeling and EDA processes, we discovered the following: 
-	Feature engineering and handling class imbalance improved model performance for all model types
-	Light GBM and Linear regression outperformed SVM and were selected as our optimal models to assist in predicting loan default.
-	Linear regression was ultimately chosen due to its simplicity, interpretability, speed, and metric performance.  
The group recommends using the LightGBM model for applications prioritizing accuracy. However, for daily use cases, a linear regression model 
is sufficient due to its comparable performance and significantly faster training time.

### Business Value:
The results and findings of our study offer business value in the following ways:
-	Operationally efficiency is promoted through the linear Regression model, as it efficiently predicts loan default and allows for quick model retraining, enabling rapid review of new applications based on current economic condition.
-	LightGBM and EDA identified key default predictors like applicant age, employment, and document submission flags, enabling targeted scrutiny of loan applications.
-	Multiple models and robustness checks in the LightGBM and Linear Regression models' formulations prevent overfitting, ensuring generalizability to future data.
-	Improve customer experience by ensuring that credit is extended to deserving customers while minimizing risk, improving satisfaction and loyalty of customers. 

### Lessons Learned:
After conducting the analysis, the group and I came away with knowledge for future projects including: 
-	Computational challenges can be overcome through a variety of techniques, including parallelism and random sampling from the training data.
-	While computationally intensive models may be marginally more accurate and useful for robustness checking, sometimes simple models like linear regression are able to capture
most of the information required and produce relatively accurate and more relevant predictions due to the speed of training.
-	While employing diverse model types enhances robustness, careful hyperparameter tuning and bias mitigation through synthetic sampling are crucial to ensure optimal performance and generalizability
