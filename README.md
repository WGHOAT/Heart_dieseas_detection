# 🫀 Heart Disease Detection

A machine learning-based system that predicts the likelihood of heart disease in a patient using clinical data. Built using Python, pandas, scikit-learn, matplotlib, and seaborn. This project involves data preprocessing, model training using Random Forest, and evaluation with accuracy and feature importance.

---

## 🚀 Objective

To develop a predictive model capable of classifying patients as having heart disease or being healthy, based on various medical parameters. The goal is to assist early diagnosis using data-driven insights.

---

## 📦 Dataset

The dataset includes a range of patient vitals and test results. Below is a description of all features:

| Feature                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `age`                  | Age in years                                                                |
| `sex`                  | 0 = Female, 1 = Male                                                        |
| `chest pain type`      | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic |
| `resting bp s`         | Resting blood pressure (in mm Hg)                                           |
| `cholesterol`          | Serum cholesterol (in mg/dl)                                                |
| `fasting blood sugar`  | 1 = >120 mg/dl, 0 = ≤120 mg/dl                                               |
| `resting ecg`          | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy                        |
| `max heart rate`       | Maximum heart rate achieved                                                 |
| `exercise angina`      | 0 = No, 1 = Yes                                                              |
| `oldpeak`              | ST depression induced by exercise                                           |
| `ST slope`             | 1 = Upsloping, 2 = Flat, 3 = Downsloping                                    |
| `target`               | 0 = Normal (No Heart Disease), 1 = Heart Disease                            |

---

## 🛠️ Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

---

## 🧠 Model Pipeline

1. **Preprocessing**:
   - Separated numerical and categorical columns.
   - Applied `StandardScaler` to numerical features.
   - Applied `OneHotEncoder` to categorical features.
   - Used `ColumnTransformer` to combine them cleanly.

2. **Train-Test Split**:
   - 80% training, 20% testing using `train_test_split`.

3. **Model**:
   - Used `RandomForestClassifier` with default parameters.
   - Trained on preprocessed training data.

4. **Evaluation**:
   - Accuracy: **95.38%**
   - Used `classification_report` and `confusion_matrix`.
   - Plotted confusion matrix and feature importance chart.

---

## 📊 Results

**Accuracy**: 95.38%

**Classification Report**:

          precision    recall  f1-score   support

       0       0.95      0.94      0.95       107
       1       0.95      0.96      0.96       131

accuracy                           0.95       238
macro avg 0.95 0.95 0.95 238
weighted avg 0.95 0.95 0.95 238

## Major Requirements
pandas 
scikit-learn 
matplotlib 
seaborn

## 🧾 Example Code Snippet
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Data loading and preprocessing...
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
# Disclimer
This Project is Built with a Intent to Learn Not to Deploy in real world and it needs way more dataset to even scatch the real thing.......
