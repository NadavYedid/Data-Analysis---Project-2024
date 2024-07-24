from flask import Flask, render_template, request
import pickle
import pandas as pd
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    en_model = pickle.load(model_file)

# Load feature names
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data with defaults
        manufactor = request.form.get('manufactor', 'unknown')
        model = request.form.get('model', 'unknown')
        Year = int(request.form.get('Year', 2000))
        Hand = float(request.form.get('Hand', 3))
        Gear = request.form.get('Gear', 'unknown')
        Color = request.form.get('Color', 'unknown')
        Km = float(request.form.get('Km', 90000))
        Prev_ownership = request.form.get('Prev_ownership', "לא מוגדר")

        # Prepare final features for prediction
        final_features = [[manufactor, model, Year, Hand, Gear, Color, Km, Prev_ownership]]

        # Convert final_features to DataFrame
        final_features_df = pd.DataFrame(final_features, columns=['manufactor', 'model', 'Year', 'Hand', 'Gear', 'Color', 'Km', 'Prev_ownership'])

        # Prepare data for prediction
        processed_features = prepare_data(final_features_df)
        
        # Align processed features with the feature names used during training
        processed_features = processed_features.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = en_model.predict(processed_features)[0]

        # Prepare output text to display on the webpage
        output_text = f"מחיר הרכב הוא: {prediction:.2f} ש״ח"

    except ValueError as ve:
        output_text = f"Value Error: {str(ve)}"
    except Exception as e:
        output_text = f"Prediction Error: {str(e)}"

    return render_template('index.html', prediction_text=output_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
