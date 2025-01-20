from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Change this to a random secret key
model = None
scaler = None
users = {"admin": "password"}  # A dictionary to store users, default user for testing

def load_model_and_scaler():
    global model, scaler
    # Load and prepare the dataset
    data_raw = pd.read_csv('data.csv')  # Ensure 'data.csv' is in the same directory as app.py
    data_cleaned = data_raw.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    data_cleaned['diagnosis'] = data_cleaned['diagnosis'].map({'M': 1, 'B': 0})

    # Prepare features and target
    X = data_cleaned.drop(columns=['diagnosis'])
    y = data_cleaned['diagnosis']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Build the ANN
    model = Sequential()
    model.add(Dense(units=30, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    
    # Check if the user exists and the password matches
    if email in users and users[email] == password:
        flash('Login successful!', 'success')
        session['logged_in'] = True  # Mark user as logged in
        return redirect(url_for('index'))  # Redirect to index page
    else:
        flash('Invalid credentials. Please try again.', 'error')
        return redirect(url_for('login_page'))

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    # Simple validation for registration
    if email in users:
        flash('Email already registered.', 'error')
    elif password != confirm_password:
        flash('Passwords do not match.', 'error')
    else:
        users[email] = password  # Store user in the dummy database
        flash('Registration successful! You can now log in.', 'success')

    return redirect(url_for('login_page'))

@app.route('/index')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login_page'))  # Ensure only logged-in users access this page
    return render_template('index.html')
@app.route('/blog')
def blog():
    return render_template('blog.html')
@app.route('/mission')
def mission():
    return render_template('mission.html')
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')
@app.route('/main')
def main():
    if 'logged_in' not in session:
        return redirect(url_for('login_page'))
    return render_template('main.html', prediction=None)  # Initialize prediction to None

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        load_model_and_scaler()  # Load the model and scaler if not already done

    # Get input data from the form
    input_data = np.array([[ 
        float(request.form['radius_mean']),
        float(request.form['texture_mean']),
        float(request.form['perimeter_mean']),
        float(request.form['area_mean']),
        float(request.form['smoothness_mean']),
        float(request.form['compactness_mean']),
        float(request.form['concavity_mean']),
        float(request.form['concave_points_mean']),
        float(request.form['symmetry_mean']),
        float(request.form['fractal_dimension_mean']),
        float(request.form['radius_se']),
        float(request.form['texture_se']),
        float(request.form['perimeter_se']),
        float(request.form['area_se']),
        float(request.form['smoothness_se']),
        float(request.form['compactness_se']),
        float(request.form['concavity_se']),
        float(request.form['concave_points_se']),
        float(request.form['symmetry_se']),
        float(request.form['fractal_dimension_se']),
        float(request.form['radius_worst']),
        float(request.form['texture_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['area_worst']),
        float(request.form['smoothness_worst']),
        float(request.form['compactness_worst']),
        float(request.form['concavity_worst']),
        float(request.form['concave_points_worst']),
        float(request.form['symmetry_worst']),
        float(request.form['fractal_dimension_worst'])
    ]], dtype=float)

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'

    # Debug print statement
    print(f"Prediction result: {result}")

    return render_template('main.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

