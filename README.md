# Diabetes Prediction using Machine Learning
## Problem Statement
Diabetes is one of the most common chronic diseases worldwide. Early prediction can help in timely treatment and lifestyle adjustments to prevent severe complications.
This project focuses on building a classification model that predicts whether a patient is diabetic (1) or not (0) based on medical attributes from the Pima Indian Diabetes dataset.

## Dataset Description
The dataset consists of medical records of female patients (≥21 years old) of Pima Indian heritage.

(Kaggle link : https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)

Features included:
- Pregnancies → Number of times pregnant
- Glucose → Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- BloodPressure → Diastolic blood pressure (mm Hg)
- SkinThickness → Triceps skin fold thickness (mm)
- Insulin → 2-Hour serum insulin (mu U/ml)
- BMI → Body Mass Index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction → A function that scores likelihood of diabetes based on family history
- Age → Age in years
- Outcome → Target variable (0 = Non-diabetic, 1 = Diabetic)

## Libraries Used
| Library       | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| **NumPy**     | Fundamental package for numerical computations in Python                |
| **Pandas**    | Data manipulation and analysis using DataFrames                         |
| **Matplotlib**| Basic plotting and data visualization                                   |
| **Seaborn**   | Statistical data visualization built on top of Matplotlib               |
| **Missingno** | Visualizations for missing data patterns                                |
| **Scikit-learn** | Machine Learning toolkit for preprocessing, model training, and evaluation |
| → train_test_split | Split dataset into training and testing sets                       |
| → GridSearchCV | Hyperparameter tuning using cross-validation                           |
| → LogisticRegression | Logistic Regression classifier                                   |
| → DecisionTreeClassifier | Decision Tree classifier                                    |
| → VotingClassifier | Ensemble method combining multiple classifiers                     |
| → RandomForestClassifier | Ensemble method using bagging of decision trees              |
| → AdaBoostClassifier | Ensemble method using boosting of weak learners                  |
| → Metrics (accuracy, recall, precision, f1-score) | Model performance evaluation |
| **XGBoost**   | Gradient boosting library optimized for speed and performance        

## Methodology
1. Data Understanding
- Explored dataset dimensions, feature types
- Checked for missing values and statistical summaries

2. Data Preprocessing
- Handled missing values
- Checked for duplicate values
- Checked for outliers using Box-plot

3. Exploratory Data Analysis (EDA)
- Univariate analysis: Distribution of individual features
- Bivariate analysis: Relationship between features and target (Outcome)
- Correlation analysis
- Handled Outliers

4. Model Development
- Split dataset into training and testing sets
- Trained multiple classification algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - AdaBoost
  - XGBoost
  - Voting Classifier (ensemble)

5. Model Evaluation
- Compared models using accuracy, precision, recall, F1-score
- Visualized performance metrics for better interpretation

6. Hyperparameter Tuning
- Optimized model parameters using GridSearchCV
- Selected the best-performing model based on evaluation results

## Results
- Achieved strong predictive performance on test data.
- Identified most important features impacting diabetes prediction.
  
## Future Work
- Experiment with deep learning models (ANN, CNN, RNN)
- Deploy the model as a web application for real-time predictions
- Incorporate additional features (diet, lifestyle, genetic data) for improved accuracy
