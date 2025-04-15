from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import pandas as pd
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
# At the start of your app.py, test if model loads correctly
try:
    model = joblib.load('xgb_final.pkl')
    print("Model loaded successfully!")
    print("Model classes:", model.classes_)
except Exception as e:
    print("Error loading model:", e)

# Load the trained model
model = joblib.load('xgb_full_features.pkl')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'Age': float(request.form['age']),
                'Experience': float(request.form['experience']),
                'Income': float(request.form['income']),
                'ZIP Code': float(request.form['zip_code']),
                'Family': float(request.form['family']),
                'CCAvg': float(request.form['ccavg']),
                'Education': float(request.form['education']),
                'Mortgage': float(request.form['mortgage']),
                'Securities Account': float(request.form['securities_account']),
                'CD Account': float(request.form['cd_account']),
                'Online': float(request.form['online']),
                'CreditCard': float(request.form['creditcard'])
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([data])
            
            # Preprocess (same as training)
            input_df['Experience'] = abs(input_df['Experience'])
            input_df['CCAvg'] = input_df['CCAvg'] * 12
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            return render_template('result.html', 
                                 prediction=prediction, 
                                 probability=f"{probability*100:.2f}%",
                                 input_data=data)
            
        except Exception as e:
            flash(f"Error processing input: {str(e)}", 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read the file
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                # Check required columns
                required_cols = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 
                                'CCAvg', 'Education', 'Mortgage', 'Securities Account', 
                                'CD Account', 'Online', 'CreditCard']
                
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    flash(f"Missing required columns: {', '.join(missing)}", 'error')
                    return redirect(request.url)
                
                # Preprocess data
                df['Experience'] = abs(df['Experience'])
                df['CCAvg'] = df['CCAvg'] * 12
                
                # Make predictions
                predictions = model.predict(df[required_cols])
                probabilities = model.predict_proba(df[required_cols])[:, 1]
                
                # Add predictions to dataframe
                df['Prediction'] = predictions
                df['Probability'] = probabilities
                
                # Save results
                results_filename = f"results_{filename}"
                results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
                
                if filename.endswith('.csv'):
                    df.to_csv(results_path, index=False)
                else:
                    df.to_excel(results_path, index=False)
                
                # Sample some results to display
                sample_results = df.head(5).to_dict('records')
                
                return render_template('batch_results.html', 
                                     sample_results=sample_results,
                                     results_file=results_filename,
                                     num_customers=len(df))
            
            except Exception as e:
                flash(f"Error processing file: {str(e)}", 'error')
                return redirect(request.url)
    
    return render_template('batch_predict.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)