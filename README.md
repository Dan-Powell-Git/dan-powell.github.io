# Portfolio

## R User Demographics

### Files
Written Report
<a href="home_Credit_files/home_credit_eda.rmd"><img src="https://img.shields.io/badge/Adobe%20Acrobat%20Reader-EC1C24.svg?style=for-the-badge&logo=Adobe%20Acrobat%20Reader&logoColor=white" alt="Open Report"></a>

R Notebook File
<a href="home_Credit_files/home_credit_eda.rmd"><img src="https://img.shields.io/badge/RStudio-4285F4?style=for-the-badge&logo=rstudio&logoColor=white" alt="Open Notebook"></a>

### Summary of Business Problem: 
The R user base is experiencing a decrease. To reverse this trend, we must identify potential users within the developer community who may be interested in adopting R

### Objective: 
Identify potential users of R to capitalize on the language's strengths and opportunities for growth.

### Data: 
- 2024 Stack Overflow Developer Survery: Developers who indicated residing in the US (~11,000 Total Sample Size)

### Methods:
- PCA and hierarchical clustering to gain a comprehensive understanding of the Stack Overflow Developer Survey.
- 80-20 Train-Test splits set before each model to test model performance
- SMOTE (Syntehtic Minority Oversampling Technique) employed on training data to reduce class bias in predictive models
- Lasso binomial regression to reduce dimensionality and identify significant variables associated with R interest.
  - 5-fold cross validation to identify optimal lambda value
- Dimensionality reduction for XGBoost and Neural Net models to improve model generalization and data collection efficiency.
- XGBoost Model to increase robustness of the analyses and produce PDP and Feature Importance charts
  - Randomly sampled training data for 7,000 observations
  - Tuned learning rate, number of trees, and depth through grid search on two levels
  - 5-fold cross validation 
- Softmax neural network to improve the model's generalization ability and capture complex, non-linear relationships within the data
  - Conducted on small sample size of 500 observations 
  - 5-fold cross validation
  - Fit model on combination of 42 different hyperparameters for decay and size to minimize error
  - AVNnet method utilized to reduce overfitting

### Results:
- Produced dendrograms for Stack Overflow Developer Survey information with clear splits along total compensation, experience, age, and other characteristics.
- Produced 3 high-performing predictive models evaluated on performacne metrics of F1 scores and accuracy.
- Identified key characteristics of R's target audience and those who may be less inclined to adopt the language

### Challenges
- Data availability: Retraining the model can only be done on the cadence of the stack overflow developer survey, which may not keep up with live trends in programming
- Data Capture: Stack overflow data is anonymous, providing a challenge on utilizing the predicitve models directly
- Potential for Overfitting: The high performance in both accuracy and F1 score may suggest that the models are overfitting the training data.

### Future Considerations:
- Improve model generalizability and robustness through additional classification models such as Adaboost and SVM
- Check for interaction effects among the variables to prevent bias in feature importance
- Develop data collection methods to complement our models and gather insights into user interest.


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
<a href="home_Credit_files/home_credit_eda.rmd"><img src="https://img.shields.io/badge/RStudio-4285F4?style=for-the-badge&logo=rstudio&logoColor=white" alt="Open Notebook"></a>

My EDA cleaned, transformed, and informed the direction of the project. The EDA highlighted several challenges with the data including:
-	Class imbalance in the target variable
-	Extreme observations in several of the independent variables
-	Redundant and noninformative independent variables
To address these limitations:
-	I performed exploratory analysis to understand the distribution of the target and independent variables, missing values, and outliers
-	I engineered new features based off correlation analyses, domain knowledge, and data insights
The dataset was refined through null imputation, class balancing with SMOTE, and the incorporation of new features inspired by insights from the EDA.

### Modeling:
<a href="home_Credit_files/Group_9_Project.ipynb"><img src="https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open Notebook"></a>

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
