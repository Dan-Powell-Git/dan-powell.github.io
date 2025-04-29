# Portfolio

## Swire Coca Cola - Capstone Project

### Files
Exploratory Data Analysis Notebook

<a href="CapStone Notebooks/EDA_Dan_Powell.html"><img src="https://img.shields.io/badge/RStudio-4285F4?style=for-the-badge&logo=rstudio&logoColor=white" alt="Open Notebook"></a>

Modeling Notebook 

<a href="CapStone Notebooks/Dan_Modeling (4).html"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Open Notebook"></a>

### Summary of Business Problem

For this project, we partnered with Swire Coca-Cola, a bottling distributor operating across the United States. Their goal was to identify which customers should be served via Red Truck—Swire’s more expensive delivery method that offers added services and helps build stronger customer relationships—versus White Truck, which is more cost-effective but limits Swire’s direct engagement with the customer. Swire is exploring the idea of implementing a threshold-based approach to determine which customers should remain on or be moved to the Red Truck channel.

### Objective

Swire provided us with data on their customer profiles, including attributes such as frequent order type (e.g., Sales Representative, Online Platform), trade channel (e.g., Wellness, School), gallons ordered in 2023 and 2024, and various other descriptive metrics. Using this data, Swire asked us to develop a new method for determining whether a customer should be assigned to the Red or White Truck channel—moving beyond the basic gallons-per-year threshold they had previously relied on.

### Methods

As my part of the project, I utilized various few analytical methods to recommend a strategy to Swire, including:

- DBSCAN Clustering: Utilized Mahalanobis distance to create clusters of customers based on their characteristics and growth metrics
  - Analyzed the proportions of various characteristics in these clusters, including proportions of customers who grew from 2023 to 2024
- Correlation Analysis: Compared the proportion of customers who grew in each cluster to the proportion utilizing various order channels
- T-Learner for Causal Inference: Utilized a T-learner to get individual treatment effects of specific order channels on customers
- Random Forest for Feature Importance: Analyzed which features contrinuted the most to average treatment effect calculated by the T-Learner
- Lasso Regression for magnitude and direction of variables: Analyzed Magnitude and direction of each of the characteristics of the customer to estimate impact on treatment effect

### Results

From our analysis, we found that Sales Representatives were the frequent order channel with the highest conditional average treatment effect across the customer base. We also estimated that if all customers not currently using the Sales Rep channel were to switch, there would be an average uplift of just over nine gallons per year. Additionally, the coefficients from our Lasso model helped identify specific customer segments—namely wellness, middle schools, and books and office—that saw the most improvement when a Sales Rep was involved.

### Challenges

We ran into a number of challenges throughout the development of this project. At first, we leaned heavily on correlation analysis to identify promising order channels for Swire to focus on. But due to the smaller dataset used for clustering and the lack of a clear uplift metric to present, we ended up pivoting to calculating average treatment effects using causal inference. After testing multiple models, we ultimately landed on the T-learner to estimate individual treatment effects. Fortunately, despite the shift in direction, everyone in the group remained engaged and proactive, and we were able to turn things around and deliver meaningful metrics that Swire can actually apply in practice.

### Future Considerations 

Based on our analysis, we recommend two actions for Swire. First, target customers under the 400-gallon threshold highlighted here by shifting them to the Sales Rep Frequent Order channel, which shows strong potential to boost their performance.

However, and perhaps more importantly, we suggest Swire adopt our method of estimating treatment effects across customers to identify which interventions — whether specific sales reps or order channels — yield positive outcomes. This approach enables Swire to proactively tailor strategies to retain and grow customers, rather than disengaging from those who could thrive with the right support.

### Lessons Learned

Through this project, I learned how to combine multiple analytical methods in sequence to generate more meaningful business insights. Rather than stopping at clustering analysis, I followed it with correlation analysis to identify which controllable factors were most associated with growth. I then took it a step further by using a T-learner to estimate individual treatment effects, and applied a Lasso model to quantify both the magnitude and direction of each characteristic’s impact on that effect. This project allowed me to bring together concepts from multiple courses I’ve taken at the University of Utah and present Swire with a well-supported, actionable solution grounded in data.

## A Pipeline for Running Gait Analysis

### Files 

PowerPoint Presentation

Write Up

Jupyter Notebook

### Summary of Business Problem 

Inspired by a recent episode of Data Skeptic where a biologist disccused attaching IMU devices to small animals as a way to observe their behavior without the scientists' phyiscal presence influncing their behavior, we wanted to apply this concept to obsevring running gait during an athlete's run. This was designed to overcome the challenges of traditional gait analysis, which are short in duration, likely influenced by the observer (self-consicousness of gait patterns more likely to occur during traditional gait analysis), and potential bias of the observer towards certain modalities. 

### Objective

With this in mind, we wanted to create a data pipeline that could be used both to capture streaming data from devices on the user - using a combination of machine learning and simple statistics to determine abnormalities in gait and to both track these abnnormailities and inform the user when they occur so they can make adjustments and avoid injury. This data would then be combined with the summary statistics for the activity to provide both a granular and hositic view of the activity for analysis and ulitimately injury prevention.

### Challenges

We initally designed the pipeline to use simulated IMU data from a study performed in 2015 and enrich this data with summary statistics from Garmin. However, due to a change in the terms of use from Garmin's API and the inability to tie the data from my runs on Strava to the study's data had us pivot directions. Instead, we utilized the GarminDB package to load in data from Garmin, which was able to provide more a more simplified metric (cadence) in a streaming fashion rather than exact IMU measurements. Even so, there is plenty of research to suggest that injury risk is heavily influenced by a lower cadence at higher speeds, meaning we could still detect injury risk whule streaming. The Garmin data was, however, able to exactly replicate the data we were looking to acquire from Strava - so the issue of summary and lap-level data was resolved completely without compromise. 

### Data

- My Garmin Activity Data from March 1 to March 20, with a 13-mile run selected for streaming simulation

### Final Constructed Pipeline

The final pipeline was constructed as such:

<img of pipeline>

### Technologies Used

- Pyspark for streaming simulation and in-flight aggregations
- Google Cloud Storage Bucket for file storage
- Pandas for dataframe manipulation
- Matplotlib for visualization creation
- Folium for Map Creation
- Google Colab for collaboration 

### Ouputs of Pipeline

- Working layered map of running route with layers for heartrate, cadence, and speed
- Simple visualizations on average heart rate over time  
- Datasets for streaming data, abnormality windows from streaming data, and summary statistics for holistic analysis for future analysis


## R User Demographics

### Files

Written Report

<a href="R-ising to the Challenge/Dan Powell Final Project Write Up.docx (1).pdf"><img src="https://img.shields.io/badge/Adobe%20Acrobat%20Reader-EC1C24.svg?style=for-the-badge&logo=Adobe%20Acrobat%20Reader&logoColor=white" alt="Open Report"></a>

R HTML Output

<a href="R-ising to the Challenge/Final-Project-Dan-Powell.html"><img src="https://img.shields.io/badge/RStudio-4285F4?style=for-the-badge&logo=rstudio&logoColor=white" alt="Open Notebook"></a>

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
