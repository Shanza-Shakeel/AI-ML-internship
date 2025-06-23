# Heart Disease Prediction 💓

## Task Objective
The goal of this project is to predict whether a person is at risk of heart disease based on their health data. Using machine learning, we analyze various features such as age, sex, cholesterol levels, blood pressure, and more to classify patients as either "Healthy" or "At Risk of Heart Disease."

This project demonstrates:
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA) to uncover trends and patterns.
- Training and evaluating classification models.
- Building an interactive Streamlit app for real-time predictions.

---

## Dataset Used
- **Dataset Name**: Heart Disease UCI Dataset  
- **Source**: [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets) 
- **Features**:
  - Numerical Features: Age, Resting Blood Pressure (`trestbps`), Cholesterol (`chol`), Maximum Heart Rate Achieved (`thalach`), etc.
  - Categorical Features: Sex, Chest Pain Type (`cp`), Fasting Blood Sugar (`fbs`), Resting ECG (`restecg`), etc.
  - Target Variable: Presence of Heart Disease (`condition`) (0 = Healthy, 1 = Disease)

The dataset contains **297 rows** and **14 columns**, with no missing values in the version used for this project.

---

## Models Applied
We experimented with the following classification models:
1. **Logistic Regression**:
   - A simple and interpretable linear model.
   - Achieved high accuracy and ROC AUC scores.
2. **Decision Tree Classifier**:
   - A non-linear model that captures complex relationships between features.
   - Useful for understanding feature importance.

Both models were trained on the preprocessed dataset and evaluated using metrics such as:
- **Accuracy**
- **ROC Curve and AUC Score**
- **Confusion Matrix**

---

## Key Results and Findings
### Model Performance
- **Logistic Regression**:
  - Accuracy: **88%**
  - ROC AUC: **94%**
- **Decision Tree**:
  - Accuracy: **85%**
  - ROC AUC: **92%**

### Important Features
The following features were identified as the most significant predictors of heart disease:
1. **Chest Pain Type (`cp`)**: Severe chest pain strongly indicates heart disease.
2. **Maximum Heart Rate Achieved (`thalach`)**: Lower heart rates during exercise are associated with higher risk.
3. **ST Depression Induced by Exercise (`oldpeak`)**: Higher ST depression indicates reduced blood flow to the heart.
4. **Number of Major Vessels Colored (`ca`)**: More blocked vessels significantly increase risk.
5. **Thalassemia Type (`thal`)**: Fixed or reversible defects are linked to heart conditions.

### Insights
- **Younger Patients**: Age plays a role, but younger patients with severe symptoms can still be at risk.
- **Cholesterol and Blood Pressure**: Elevated levels are strong indicators of heart disease.
- **Exercise-Induced Angina**: Presence of angina during exercise is a key warning sign.

---

## How to Run the App
1. Clone this repository:
   ```bash
   git clone https://github.com/Shanza-Shakeel/AI-ML-internship/Heart-Disease-Prediction.git 
   cd Heart-Disease-Prediction