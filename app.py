from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('./models/credit_approval_model.joblib')
scaler = joblib.load('./models/scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    age = int(request.form['age'])
    income = float(request.form['income'])
    credit_score = int(request.form['credit_score'])
    debt_ratio = float(request.form['debt_ratio'])
    loan_amount = float(request.form['loan_amount'])

    # Create the input array for the model
    input_data = np.array([[age, income, credit_score, debt_ratio, loan_amount]])

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    
    # Interpret the prediction
    output = 'Approved' if prediction[0] == 1 else 'Not Approved'
    
    return render_template('index.html', prediction_text=f'Loan Status: {output}')

if __name__ == "__main__":
    app.run(debug=True)
