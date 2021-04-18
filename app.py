# import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
from collections import OrderedDict
from models.model import handle_missing_values, map_dependents, handle_cat_data

# app name
app = Flask(__name__)

# load the saved model
def load_model():
    return pickle.load(open('loan_model.pkl','rb'))

# home page
@app.route('/')
def home():
    return render_template('index.html')

# predict the results and return it to frontend

def create_example(values):
    example_dict = OrderedDict()
    property_dict = OrderedDict()
    cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History']

    property_area = {'Property_Area_0':'Rural', 'Property_Area_1':'Urban', 'Property_Area_2': 'Semiurban'}

    cat = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Property_Area']

    num = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Loan_Status']

    for col, value in zip(cols, values):
        if col in num:
            example_dict[col] = float(value)
        else:
            example_dict[col] = str(value)
    
    example_df = pd.DataFrame(example_dict, index=[0])

    handle_missing_values(example_df)
    map_dependents(example_df)
    example_df = handle_cat_data(example_df)
    
    for col in property_area.keys():
        property_dict[col] = 1 if property_area[col] == values[-1] else 0
    
    property_df = pd.DataFrame(property_dict, index=[0])

    example = pd.concat([example_df, property_df], axis=1)

    
    return example
    

@app.route('/predict', methods=['POST'])
def predict():

    labels = ['Approved', 'Rejected']

    features = [x for x in request.form.values()]

    example = create_example(features)
    
    model = load_model()
    prediction = model.predict(example)

    result = labels[prediction[0]]

    return render_template('index.html', output='Your loan is {}'.format(result))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)