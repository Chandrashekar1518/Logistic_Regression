
import streamlit as st
import pickle
import pandas as pd

# Load the trained model and columns
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Define the prediction function
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Create a dataframe with user inputs
    input_data = pd.DataFrame([{
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }])
    # Apply same preprocessing as training
    input_data['Embarked'] = input_data['Embarked'].fillna('UNKNOWN')

    # One-hot encode user inputs (same as training)
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)

    # Add any missing columns that the model expects
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match training
    input_data_encoded = input_data_encoded[model_columns]

    # Predict using the model
    prediction = model.predict(input_data_encoded)[0]
    return "Survived" if prediction == 1 else "Did Not Survive"

# Streamlit app UI
def main():
    st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

    st.markdown("""
        <div style="background-color: #add8e6; padding: 10px; border-radius: 10px;">
        <h2 style="text-align: center; color: black;">Titanic Survival Prediction (Logistic Regression)</h2>
        </div>
    """, unsafe_allow_html=True)

    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex", ['Male', 'Female'])
    Age = st.slider("Age", 0, 100, 30)
    SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10)
    Parch = st.number_input("Parents/Children Aboard", 0, 10)
    Fare = st.number_input("Ticket Fare", 0.0, 600.0, 32.2)
    Embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S', 'UNKNOWN'])


    if st.button("Predict"):
        result = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
        if result == "Survived":
            st.markdown(f"""
            <div style='background-color:#d4edda;padding:15px;border-radius:10px'>
            <h3 style='color:green;text-align:center;'>Prediction: {result}</h3>
           </div>
           """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div style='background-color:#f8d7da;padding:15px;border-radius:10px'>
            <h3 style='color:red;text-align:center;'>Prediction: {result}</h3>
            </div>
            """, unsafe_allow_html=True)
        
# Run the app
if __name__ == '__main__':
    main()
