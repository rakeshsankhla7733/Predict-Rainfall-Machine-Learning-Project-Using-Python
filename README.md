# Predict-Rainfall-Machine-Learning-Project-Using-Python
Various classification models were developed and evaluated to predict rainfall in Sydney using  decision trees, random forests and gradient boosting.
# Rainfall Prediction Project

This repository contains a Jupyter notebook for predicting rainfall using various machine learning models. The project involves data preprocessing, exploratory data analysis, model training, evaluation, and comparison.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to predict rainfall using machine learning techniques. The notebook includes steps for data preprocessing, model training, evaluation, and comparison of various models.

## Dataset
The dataset used for this project contains historical rainfall data. It includes features such as temperature, humidity, and other meteorological variables.

## Installation
To run this project, you need to have Python and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost


## Usage
### 1.Clone this repository:
git clone https://github.com/rakeshsankhla7733/rainfall-prediction.git

### 2.Navigate to the project directory:
cd rainfall-prediction

### 3.Open the Jupyter notebook:
jupyter notebook Rain_fall_interview_prepration.ipynb

### 4.Run the cells in the notebook to execute the code.

## Models and Evaluation
The notebook includes the following models:

- Logistic Regression
- K-Nearest Neighbors
- Linear Discriminant Analysis
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost

- The models are evaluated based on accuracy, ROC AUC score, recall, and precision.

## Results
The Random Forest model performed the best with an accuracy score of 82.485%. Detailed comparison and evaluation metrics for all models are provided in the notebook.

## Future Work
To further improve the model's performance, the following steps can be taken:

- Perform hyperparameter tuning using techniques such as Grid Search or Random Search.
- Use cross-validation to ensure the model generalizes well to unseen data.
- Experiment with different feature engineering techniques to create new features.
- Incorporate additional data sources if available.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


### Visualizations

To enhance the notebook with visualizations, you can add the following plots:

1. **Correlation Heatmap:**
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    ```

2. **Feature Importance (for Random Forest):**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # Assuming X_train and y_train are defined
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.importance, y=feature_importances.index)
    plt.title('Feature Importance')
    plt.show()
    ```

3. **ROC Curves for All Models:**
    ```python
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(12, 8))

    models = {
        "Logistic Regression": logistic_regression,
        "K-Nearest Neighbors": knn,
        "Linear Discriminant Analysis": lda,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "AdaBoost": ada_boost,
        "XGBoost": xg_boost
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()
    ```

