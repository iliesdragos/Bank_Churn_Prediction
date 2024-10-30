# Bank Churn Prediction

![R](https://img.shields.io/badge/R-Programming-blue)
![RStudio](https://img.shields.io/badge/RStudio-IDE-blue)

A project to analyze and predict customer churn (exit) in a bank using various machine learning techniques. This project was created in R and explores the application of different classification algorithms to maximize the accuracy of predictions, particularly focusing on identifying customers likely to churn.


## Dataset

The dataset consists of customer information from a bank in the U.S., sourced from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data). Key columns used include:

- **CreditScore**: Customer’s credit score
- **Geography**: Country of residence
- **Gender**: Customer's gender
- **Age**: Age of the customer
- **Tenure**: Years the customer has been with the bank
- **Balance**: Average account balance
- **NumOfProducts**: Number of products used by the customer
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No)
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No)
- **EstimatedSalary**: Estimated annual salary
- **Exited**: Churn indicator (1 = Yes, 0 = No)

## Objectives

1. Identify key factors that influence customer churn in the banking sector.
2. Compare different machine learning algorithms to find the best-performing model for predicting churn.
3. Analyze the impact of dataset balancing on model performance, specifically on **specificity** and **sensitivity**.

## Exploratory Data Analysis

Preliminary visualizations were created to understand the distribution of key variables, including:

- Distribution of `Exited` (churned vs. non-churned customers)
- Churn rates by **Gender**
- Impact of **CreditScore** and **Age** on churn

## Methodology

The following machine learning algorithms were applied to predict churn:

1. **Logistic Regression**
2. **Naive Bayes**
3. **Decision Trees**
4. **Bagging** (ensemble of decision trees)
5. **Random Forest** (with hyperparameter tuning)

Each model was evaluated on metrics including specificity, sensitivity, and AUC-ROC. Initial models were built on an imbalanced dataset, then later retrained on a balanced dataset using the `ROSE` package in R.

### Model Performance Comparison (Imbalanced Dataset)

| Model            | Specificity | Sensitivity | AUC-ROC |
|------------------|-------------|-------------|---------|
| Logistic Regression | 22.55% | 96.74% | 0.7683 |
| Naive Bayes         | 28.92% | 97.74% | 0.8141 |
| Decision Tree (basic) | 41.99% | 97.53% | 0.7686 |
| Decision Tree (no restrictions) | 53.76% | 92.55% | 0.8307 |
| Bagging           | 50.65% | 94.77% | 0.8405 |
| Random Forest     | 47.06% | 96.94% | 0.8663 |

### Model Performance Comparison (Balanced Dataset)

| Model            | Specificity | Sensitivity | AUC-ROC |
|------------------|-------------|-------------|---------|
| Logistic Regression | 69.12% | 73.80% | 0.7744 |
| Naive Bayes         | 65.20% | 80.54% | 0.81   |
| Decision Tree (pruned, cp=0.02) | 75.65% | 74.17% | 0.7910 |
| Bagging           | 60.95% | 89.07% | 0.8430 |
| Random Forest     | 62.91% | 90.41% | 0.8587 |

## Key Findings

1. **Most Influential Factors**:
   - **Age**: Older customers are more likely to churn.
   - **NumOfProducts**: Customers with fewer products have higher churn rates.
   - **IsActiveMember**: Active members are less likely to churn.
   - **Geography**: Customers in Germany showed higher churn rates than those in France or Spain.

2. **Effectiveness of Balancing**:
   - Balancing the dataset improved specificity across models, aiding in identifying churners more accurately.
   - However, it resulted in a slight decrease in sensitivity, which is a common trade-off.

3. **Best Performing Model**:
   - The **Random Forest** model on the balanced dataset achieved the best overall performance with a high AUC-ROC and balanced sensitivity and specificity.

## Running the Project

To reproduce this project:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/Bank_Churn_Prediction.git
   ```

2. Open proiect.R in RStudio.

3. Install required packages:
   ```bash
   install.packages(c("ROSE", "caret", "rpart", "randomForest"))
   ```

4. Run each section of the code in proiect.R to see the data analysis, modeling, and evaluation results.

## Conclusion

This analysis demonstrated the importance of using a balanced dataset for churn prediction. Key predictors of churn were identified, providing actionable insights for banks to develop targeted retention strategies.

## Future Improvements

- **Dynamic Feature Selection**: Experiment with additional feature selection techniques.
- **Advanced Balancing Techniques**: Test other balancing methods like SMOTE to improve model performance.
- **Model Deployment**: Implement a production-ready model with live churn predictions.


