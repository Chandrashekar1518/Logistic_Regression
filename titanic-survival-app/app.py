
from flask import Flask, request, jsonify,render_template
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load the trained model and columns
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Prediction function
def predict_survival(data):
    # Create DataFrame from incoming JSON
    input_data = pd.DataFrame([data])

    # Fill missing categorical values
    input_data['Embarked'] = input_data['Embarked'].fillna('UNKNOWN')
    input_data['Sex'] = input_data['Sex'].fillna('UNKNOWN')

    # One-hot encode
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)

    # Add missing columns
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns
    input_data_encoded = input_data_encoded[model_columns]

    # Predict
    prediction = model.predict(input_data_encoded)[0]
    result = "Survived " if prediction == 1 else "Did Not Survive 💀"
    return result

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Titanic Survival API is running!"})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        prediction = predict_survival(data)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run app
if __name__ == '__main__':
    app.run(debug=True)
