# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
def load_model_and_scaler():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

# Define the main function for the app
def main():
    # Set page title and description
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="wide")
    st.title("Heart Disease Prediction App 💓")
    st.markdown("""
    ### Welcome to the Heart Disease Prediction App!
    This app predicts whether a patient is at risk of heart disease based on their health data. 
    Simply fill in the details in the sidebar and click "Predict" to get the result.
    """)

    # Sidebar for user input
    st.sidebar.header("Patient Information 📋")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["No Pain", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Severe Pain"])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=50, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "Abnormal ST-T Wave", "Probable or Definite Hypertrophy"])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert categorical inputs to numeric values
    sex = 1 if sex == "Male" else 0
    cp_map = {"No Pain": 0, "Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Severe Pain": 4}
    cp = cp_map[cp]
    fbs = 1 if fbs == "Yes" else 0
    restecg_map = {"Normal": 0, "Abnormal ST-T Wave": 1, "Probable or Definite Hypertrophy": 2}
    restecg = restecg_map[restecg]
    exang = 1 if exang == "Yes" else 0
    slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
    slope = slope_map[slope]
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal = thal_map[thal]

    # Create a DataFrame from the inputs
    patient_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    patient_df = pd.DataFrame([patient_data])

    # Predict button
    if st.sidebar.button("Predict"):
        try:
            # Load the model and scaler
            model, scaler = load_model_and_scaler()

            # Scale the data
            scaled_data = scaler.transform(patient_df)

            # Make prediction
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)[0][1]

            # Display results in the main area
            st.subheader("Prediction Results 📊")
            if prediction[0] == 1:
                st.error("**Prediction: Heart Disease 😔**")
            else:
                st.success("**Prediction: Healthy 😊**")
            
            # Display probability with a progress bar
            st.write(f"**Probability of Heart Disease:** {probability * 100:.1f}%")
            st.progress(int(probability * 100))

            # Highlight key risk factors
            st.subheader("Key Risk Factors 🔍")
            risk_factors = {
                "Age": "Higher risk with older age." if age > 50 else "Age is not a significant risk factor.",
                "Chest Pain Type": "Severe chest pain is a strong indicator of heart disease." if cp >= 3 else "Chest pain type is not severe.",
                "Cholesterol": "High cholesterol (>200 mg/dL) increases risk." if chol > 200 else "Cholesterol level is within normal range.",
                "Exercise-Induced Angina": "Presence of angina during exercise indicates higher risk." if exang == 1 else "No exercise-induced angina.",
                "ST Depression": "Significant ST depression indicates reduced blood flow." if oldpeak > 1.5 else "ST depression is minimal."
            }
            for factor, explanation in risk_factors.items():
                st.markdown(f"- **{factor}:** {explanation}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # About Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the App ℹ️")
    st.sidebar.markdown("""
    This app uses a machine learning model trained on the Heart Disease UCI dataset to predict the likelihood of heart disease.
    The model considers various health parameters such as age, blood pressure, cholesterol, and more.
    """)

if __name__ == "__main__":
    main()