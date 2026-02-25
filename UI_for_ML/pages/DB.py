import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np

# ---------------- MODEL LOADING (CACHE SAFE) -
# ---------------
@st.cache_resource
def load_model():
    model_load = pickle.load(open(r'C:\Users\KIIT0001\Documents\GitHub\Mini_project_ML\random_forest_model.sav', 'rb'))
    return model_load

#def diabetes_prediction(input_data):
    model_load = load_model()
    input_data_as_numpy_array=np.asarray(input_data)

# reshape the array as we are predicting for one instance   
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=model_load.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==0):
      return('The person is not at risk of diabetes')
    else: return('The person is at risk of diabetes')


def main():
    st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

    # --- Custom CSS ---
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

            html, body, [class*="css"]  {
                font-family: 'Outfit', sans-serif;
            }

            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }

            h1, h2, h3 {
                color: #2c3e50;
                font-weight: 700;
            }

            [data-testid="stForm"] {
                background: rgba(255, 255, 255, 0.85);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
            }

            div.stButton > button {
                background: linear-gradient(45deg, #FF6B6B, #EE5D5D);
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 12px;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧬 Diabetes Prediction System")

    # -------- Load Dataset --------
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, "required_files", "Diabetes_dataset_2.csv")

        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            st.subheader("Enter Patient Details")

            # -------- FORM --------
            with st.form("entry_form", clear_on_submit=True):

                c1, c2, c3 = st.columns(3)

                with c1:
                    age = st.number_input("Age", 0, 120, 25)
                    bmi = st.number_input("BMI", 0.0, 60.0, 22.5)
                    glucose_level = st.number_input("Glucose Level", 0.0, 300.0, 100.0)

                with c2:
                    physical_activity_level = st.selectbox("Physical Activity Level", ["low", "moderate", "high"])
                    family_history = st.selectbox("Family History", [0, 1])

                with c3:
                    smoker = st.selectbox("Smoker", [0, 1])

                submitted = st.form_submit_button("Predict & Save", type="primary")

                # -------- PREDICTION --------
                if submitted:

                    # Encode categorical
                    activity_map = {"low": 0, "moderate": 1, "high": 2}
                    physical_activity_encoded = activity_map[physical_activity_level]

                    # Prepare input
                    model_input = pd.DataFrame([{
                        'age': age,
                        'bmi': bmi,
                        'glucose_level': glucose_level,
                        'physical_activity_level': physical_activity_encoded,
                        'family_history': family_history,
                        'smoker': smoker,
                    }])

                    # Prediction
                    # -------- Prediction --------
                    model_load = load_model()
                    prediction = model_load.predict(model_input)[0]

# Probability of class 1 (At Risk)
                    risk_prob = model_load.predict_proba(model_input)[0][1]
                    safe_prob = 1 - risk_prob

                    st.divider()
                    st.subheader("Prediction Result")

                    if prediction == 1:
                          st.error("⚠️ Patient is AT RISK of Diabetes")
                          st.metric("Risk Probability", f"{risk_prob*100:.2f}%")
                    else:
                          st.success("✅ Patient has LOW Diabetes Risk")
                          st.metric("Safety Confidence", f"{safe_prob*100:.2f}%")


                    # Save record
                    save_data = model_input.copy()
                    save_data["prediction"] = prediction
                    save_data.to_csv(data_path, mode='a', header=False, index=False)

                    st.toast("Prediction generated & stored!", icon="🤖")

            # -------- Preview --------
            st.divider()
            st.subheader("Dataset Preview")
            st.dataframe(df.tail(5), use_container_width=True)

        else:
            st.error("Dataset file not found!")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    load_model()  # Preload model for caching
    main()
