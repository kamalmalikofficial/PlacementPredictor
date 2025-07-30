from flask import Flask
from flask import render_template, request
import numpy as np 
import pandas as pd
import pickle

app = Flask(__name__)
with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    iq= request.form.get('iq')
    cgpa= request.form.get('cgpa')
    project= request.form.get('projet')
    internex= request.form.get('internex')

    input_data = pd.DataFrame([[iq, cgpa, internex, project]],
        columns=['IQ', 'CGPA', 'Internship_Experience', 'Projects_Completed']
    )

    prediction = model.predict(input_data)
    return str(prediction[0]) 

if __name__ == '__main__':
    app.run(debug=True)
