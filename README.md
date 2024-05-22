# Crop Recommendation System

## Overview
This project involves the development of a Crop Recommendation System using machine learning techniques. The system is designed to recommend the most suitable crop to cultivate based on various soil and weather conditions. The dataset used includes information on nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall for different crops.

## Dataset
The dataset contains the following columns:
- `N`: Nitrogen content in the soil
- `P`: Phosphorus content in the soil
- `K`: Potassium content in the soil
- `temperature`: Temperature in Celsius
- `humidity`: Relative humidity in %
- `ph`: pH value of the soil
- `rainfall`: Rainfall in mm
- `label`: The type of crop

## Exploratory Data Analysis (EDA)
I performed an extensive exploratory data analysis to understand the distribution and characteristics of the data. The key steps involved:
1. Checking for missing values.
2. Analyzing the statistical properties of the dataset.
3. Visualizing the distribution of each feature using histograms and KDE plots.
4. Plotting box plots to detect potential outliers for each feature across different crops.

## Outlier Detection
I implemented a function to detect outliers based on the Interquartile Range (IQR) method. Outliers were evaluated for each feature per crop, and it was concluded that no significant outliers needed to be removed.

## Data Preprocessing
The features were scaled using `StandardScaler` to standardize the dataset before training the models.

## Model Training
I trained several machine learning models using GridSearchCV to find the best hyperparameters. The models evaluated included:
1. Decision Tree Classifier
2. Support Vector Machine (SVM)
3. Random Forest Classifier
4. K-Nearest Neighbors Classifier (KNN)
5. XGBoost Classifier
6. Gradient Boosting Classifier
7. AdaBoost Classifier

## Model Evaluation
The performance of each model was evaluated using accuracy scores, and the best performing models were:
- Random Forest Classifier
- XGBoost Classifier

Additionally, a Bagging Classifier was implemented with the Random Forest as the base estimator to further enhance the model's performance.

## Results
The Random Forest model achieved the highest accuracy of 99.82% on the test set. The Bagging Classifier with Random Forest also performed remarkably well with an accuracy of 99.64%.

## Confusion Matrix and Classification Report
Confusion matrices and classification reports were generated to provide detailed insights into the model performance for each crop class.

## Conclusion
The Crop Recommendation System effectively predicts the best crop to cultivate based on various soil and weather conditions. The models were fine-tuned to achieve high accuracy, making them reliable for practical use in agriculture.
## License
This project is licensed under the MIT License.
