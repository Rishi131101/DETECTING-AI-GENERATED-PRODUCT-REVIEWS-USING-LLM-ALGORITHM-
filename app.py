from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the model and vectorizer
model = joblib.load("etc_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")  # Your frontend upload page

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('csv')  # Ensure the 'name' attribute of the input is 'csv'
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Read CSV into DataFrame
    df = pd.read_csv(file)

    # Validate required column
    if 'text' not in df.columns:
        return jsonify({'error': "CSV must have a 'text' column"}), 400

    # Make predictions
    X = vectorizer.transform(df['text'])
    predictions = model.predict(X)

    # Add prediction column
    df['prediction'] = predictions

    # Map 0 -> Human Generated, 1 -> AI Generated
    df['generated'] = df['prediction'].apply(lambda x: 'Human Generated' if x == 0 else 'AI Generated')

    # Optional: drop the numeric prediction column
    df = df.drop(columns=['prediction'])

    # Return results as JSON
    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)

