
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('salaryindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        education_level = float(request.form['education_level'])
        job_title = float(request.form['job_title'])
        years_of_experience = float(request.form['years_of_experience'])

        features = np.array([[age, gender, education_level, job_title, years_of_experience]])
        features_scaled = scaler.transform(features)
        predicted_salary = model.predict(features_scaled)[0]

        return f"<h3>💰 Predicted Salary: ₹{predicted_salary:,.2f}</h3>"

    except Exception as e:
        return f"<h3 style='color:red;'>Error: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
