from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask_cors import CORS
import logging

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load model and label encoders once at app startup
try:
    model = joblib.load("rf_smote_model.pkl")
    le_dict = joblib.load("label_encoders.pkl")
except FileNotFoundError:
    logging.error("Model or label encoders file not found. Ensure 'rf_smote_model.pkl' and 'label_encoders.pkl' are in the same directory.")
    #print("Error: Model or label encoders file not found. Ensure 'rf_smote_model.pkl' and 'label_encoders.pkl' are in the same directory.")
    # Exit or handle the error gracefully
    exit()

def preprocess(input_data: pd.DataFrame, le_dict):
    # Convert datetime columns
    input_data["TransactionDate"] = pd.to_datetime(input_data["TransactionDate"])
    input_data["PreviousTransactionDate"] = pd.to_datetime(input_data["PreviousTransactionDate"])
    
    # Rename IP_Address column if present
    #if "IP_Address" in input_data.columns:
    #    input_data = input_data.rename(columns={"IP_Address": "IP Address"})
        
    # Encode categorical variables with pre-fitted label encoders
    for col, le in le_dict.items():
        if col in input_data.columns:
            # Handle unknown categories by mapping to a default value (e.g., the first class)
            input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_data[col] = le.transform(input_data[col])
    
    # Select features in correct order
    feature_cols = ['TransactionAmount', 'TransactionDate', 'Location', 'IP Address', 'MerchantID', 'PreviousTransactionDate']
    
    # Ensure all required columns are present before selecting
    present_feature_cols = [col for col in feature_cols if col in input_data.columns]
    input_processed = input_data[present_feature_cols]

    return input_processed

# Route for the home page to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction request from the form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess the input data
        processed_df = preprocess(input_df, le_dict)
        
        # Run prediction
        prediction = model.predict(processed_df)[0]
        
        # Interpret result
        result = "Fraud" if prediction == 1 else "Non-Fraud"

        # Return a JSON response
        return jsonify({"prediction": result})

    except Exception as e:
        # Return a JSON error response
        logging.exception("Prediction error:")  # Log the full stack trace
        return jsonify({"error": str(e)}), 500  # Return 500 for server errors
        #return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

# Note: Ensure that 'index.html' exists in the 'templates' directory for rendering the form.