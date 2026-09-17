# project-3
test repo 3
this repo is created by Mohammad Jaradat
print("Hello")
# Breast Cancer Classification Using Machine Learning

## 📌 Project Overview

This project applies several Machine Learning classification algorithms to predict whether a breast tumor is **Benign** or **Malignant**.

The project uses the Breast Cancer dataset and compares multiple classification models to determine their performance. The models are trained, optimized using **GridSearchCV**, and evaluated using **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix**.

The project also includes **feature importance visualization** for models that support it.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Load and preprocess the breast cancer dataset.
* Convert the target labels into numerical values.
* Split the dataset into training and testing sets.
* Standardize the numerical features.
* Train multiple Machine Learning classification models.
* Perform hyperparameter tuning using GridSearchCV.
* Compare the performance of the models.
* Select the model with the highest test accuracy.
* Visualize important features when supported by the selected model.
* Evaluate the final model using several classification metrics.
* Display the confusion matrix.

---

## 📊 Dataset

The dataset contains information about breast tumor characteristics.

The target variable is:

```text
Diagnosis
```

The target contains two classes:

```text
B = Benign
M = Malignant
```

The dataset contains **569 samples** and **30 numerical features**.

The features describe measurements related to the characteristics of the cell nuclei, such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave Points
* Symmetry
* Fractal Dimension

These measurements are provided in different forms, including:

* Mean values
* Standard error values
* Worst/largest values

---

# 🔄 Project Workflow

The project follows this Machine Learning workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Label Encoding
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Hyperparameter Tuning
   ↓
Train Multiple Models
   ↓
Model Comparison
   ↓
Select Best Model
   ↓
Feature Importance
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Confusion Matrix
```

---

## 1. 📥 Import Libraries

The project starts by importing the required Python libraries:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

Additional Scikit-learn models are imported:

```python
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
```

The project also uses Matplotlib and Seaborn for visualization.

---

## 2. 📂 Load the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv('breast_cancer_data.csv')
```

The dataset is stored in a Pandas DataFrame called `df`.

---

## 3. 🔤 Encode the Target Variable

The `Diagnosis` column contains categorical values (`B` and `M`).

Since Machine Learning models require numerical target values, `LabelEncoder` is used:

```python
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])
```

The labels are converted into numerical values:

```text
B → 0
M → 1
```

Therefore:

```text
0 = Benign
1 = Malignant
```

---

## 4. 🎯 Separate Features and Target

The target column is separated from the input features:

```python
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
```

Where:

* `X` contains the input features.
* `y` contains the target labels.

The project then checks the number of samples, number of features, and class distribution.

---

## 5. ✂️ Train-Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y
)
```

The parameters mean:

* `test_size=0.3` → 30% of the data is used for testing.
* `random_state=1` → makes the split reproducible.
* `stratify=y` → keeps a similar class distribution in training and testing sets.

---

## 6. 📏 Feature Scaling

The numerical features are standardized using `StandardScaler`:

```python
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
```

The scaler is fitted only on the training data and then applied to both training and testing data.

This helps models such as Logistic Regression, Perceptron, and SVM work with features on comparable scales.

---

# 🤖 Machine Learning Models

The project compares five Machine Learning classification algorithms:

### 1. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of a sample belonging to a class.

Hyperparameters tested:

```python
C = [0.1, 1, 10]
solver = ['liblinear']
penalty = ['l1', 'l2']
```

---

### 2. Perceptron

The Perceptron is a simple linear classification algorithm based on learning a decision boundary between classes.

Hyperparameters tested include:

```python
alpha = [0.0001, 0.01]
penalty = ['l1', 'l2', None]
eta0 = [0.1, 1.0]
```

---

### 3. Decision Tree

Decision Tree classification uses a tree-like structure to make decisions based on feature values.

Hyperparameters tested include:

```python
criterion = ['gini', 'entropy']
max_depth = [None, 5, 10]
min_samples_split = [2, 5]
```

---

### 4. Support Vector Machine (SVM)

SVM attempts to find a decision boundary that separates the classes.

The project tests:

```python
C = [0.1, 1, 10]
kernel = ['linear', 'rbf']
gamma = ['scale', 'auto']
```

---

### 5. Random Forest

Random Forest combines multiple decision trees to perform classification.

The project tests:

```python
n_estimators = [50, 100]
max_features = ['sqrt', 'log2']
max_depth = [None, 10]
```

---

# ⚙️ Hyperparameter Tuning

The project uses **GridSearchCV** to find suitable hyperparameter combinations.

For example:

```python
grid_lr = GridSearchCV(
    LogisticRegression(),
    param_lr,
    cv=5
).fit(X_train_scaled, y_train)
```

`cv=5` means that **5-fold cross-validation** is used during the grid search.

The same approach is applied to the different models.

The best estimator from each GridSearch is stored and later compared.

---

# 🏆 Model Selection

After training the models, the project calculates the test accuracy for each optimized model:

```python
test_accuracy = grid.score(X_test_scaled, y_test)
```

The results are stored in a DataFrame:

```python
comparison_df = pd.DataFrame(results)
```

The models are then sorted according to test accuracy:

```python
comparison_df = comparison_df.sort_values(
    by='Test Accuracy',
    ascending=False
)
```

The first model in the sorted DataFrame is selected as the model with the highest measured test accuracy:

```python
best_model_object = comparison_df.iloc[0]['ModelObject']
best_model_name = comparison_df.iloc[0]['ModelName']
```

---

# 📊 Feature Importance

The project checks whether the selected model provides a `feature_importances_` attribute:

```python
if hasattr(best_model_object, 'feature_importances_'):
```

If available, the feature importance values are extracted:

```python
importances = best_model_object.feature_importances_
```

A DataFrame is created containing:

```text
Feature
Importance
```

The features are sorted from highest to lowest importance, and the **top 10 features** are visualized using a bar plot.

This helps identify which features contributed most to the selected model's predictions.

---

# 🔮 Prediction

After selecting the model, predictions are made on the test set:

```python
y_pred = best_model_object.predict(X_test_scaled)
```

`y_pred` contains the predicted class for each test sample.

The predictions are then compared with the actual values in `y_test`.

---

# 📈 Model Evaluation

The final model is evaluated using four metrics.

### Accuracy

```python
accuracy_score(y_test, y_pred)
```

Accuracy measures the proportion of correctly classified samples.

### Precision

```python
precision_score(y_test, y_pred)
```

Precision measures how many samples predicted as positive are actually positive.

### Recall

```python
recall_score(y_test, y_pred)
```

Recall measures how many of the actual positive samples were correctly identified.

### F1-Score

```python
f1_score(y_test, y_pred)
```

F1-Score combines Precision and Recall into a single metric.

The project prints all four metrics:

```text
Accuracy
Precision
Recall
F1 Score
```

---

# 📉 Confusion Matrix

The project also generates a confusion matrix:

```python
cm = confusion_matrix(y_test, y_pred)
```

The confusion matrix is visualized using Seaborn:

```python
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)
```

The matrix shows the relationship between:

* Actual Benign
* Actual Malignant
* Predicted Benign
* Predicted Malignant

This provides a more detailed view of the model's classification results.

---

# 🌳 Additional Model: Gradient Boosting

The project also contains an additional experiment using **Gradient Boosting Classifier**.

```python
from sklearn.ensemble import GradientBoostingClassifier
```

GridSearchCV is used to tune:

```python
n_estimators = [100, 200]
learning_rate = [0.05, 0.1]
max_depth = [3, 4]
```

The best Gradient Boosting model is obtained using:

```python
best_gb = grid_gb.best_estimator_
```

It is then evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Gradient Boosting feature importance is also visualized using:

```python
best_gb.feature_importances_
```

and the top 10 features are displayed.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas** – data loading and manipulation
* **NumPy** – numerical operations
* **Scikit-learn** – Machine Learning algorithms and evaluation
* **Matplotlib** – visualization
* **Seaborn** – statistical visualization
* **Google Colab / Jupyter Notebook** – development environment

---

# 📁 Project Files

```text
ML-Project/
│
├── ML_Project.ipynb
├── breast_cancer_data.csv
└── README.md
```

### `ML_Project.ipynb`

Contains the complete Machine Learning implementation, including preprocessing, model training, hyperparameter tuning, model selection, prediction, and evaluation.

### `breast_cancer_data.csv`

Contains the breast cancer data used by the Machine Learning models.

### `README.md`

Contains the project documentation and explanation.

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the notebook

Open:

```text
ML_Project.ipynb
```

using Google Colab or Jupyter Notebook.

### 3. Make sure the dataset is in the same directory

```text
breast_cancer_data.csv
```

### 4. Run the notebook

Run the cells from top to bottom.

The notebook will:

1. Load the dataset.
2. Encode the target variable.
3. Split the data.
4. Scale the features.
5. Tune the models.
6. Compare the models.
7. Select the model with the highest measured test accuracy.
8. Generate predictions.
9. Calculate evaluation metrics.
10. Display the confusion matrix.
11. Visualize feature importance when supported.

---

# 📌 Summary

This project demonstrates a complete Machine Learning classification workflow for breast cancer diagnosis. Multiple classification algorithms are trained and optimized using **GridSearchCV**. Their performance is evaluated using standard classification metrics, and the selected model is further analyzed using a confusion matrix and feature importance visualization when available.

The project provides practical experience with **data preprocessing, feature scaling, model training, hyperparameter tuning, model comparison, prediction, and evaluation**.

