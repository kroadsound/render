from flask import Flask, request, jsonify, render_template
import pandas as pd

import joblib
from utils import preprocessor

app = Flask(__name__)
model = joblib.load(open('model.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    input = request.form['text']
    predicted_sentiment = model.predict(pd.Series(input))[0]
    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template('index.html', sentiment=f'Predicted sentiment of "{input}" is {output}.')


if __name__ == "__main__":
    app.run(debug=True)