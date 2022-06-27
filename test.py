import pandas as pd
import inquirer
import numpy as np
from flask import Flask, render_template, request
import pickle
import jinja2


app = Flask(__name__)
  
with open("model_pickle","rb")as f:
    saved_model =pickle.load(f)
    # <!-- Year	Present_Price	Kms_Driven	Fuel_Type	Seller_Type	Transmission	Owner    -->
car=pd.read_csv("car data.csv")
car=car.dropna()


@app.route('/')
# ‘/’ URL is bound with hello_world() function.

def hello_world():

    return render_template('index.html',)

# companies=company,fuel_type=fuel_type,Seller_Type=Seller_Type, Transmission= Transmission,Owner=Owner

@app.route('/predict', methods=['POST'])
def predict():
    Year=int(request.form.get('Year'))
    Present_Price=float(request.form.get('Present_Price'))
    Kms_Driven=int(request.form.get('Kms_Driven'))
    Fuel_Type=int(request.form.get('Fuel_Type'))
    Seller_Type=int(request.form.get('Seller_Type'))
    Transmission=int(request.form.get('Transmission'))
    Owner=int(request.form.get('Owner'))
    print(Year,	Present_Price,	Kms_Driven,	Fuel_Type,	Seller_Type,	Transmission,	Owner )
    prediction=saved_model.predict(pd.DataFrame([[Year,Present_Price,Kms_Driven,Fuel_Type,	Seller_Type,Transmission,Owner]], columns=['Year',	'Present_Price',	'Kms_Driven',	'Fuel_Type',	'Seller_Type',	'Transmission',	'Owner']))
    # print(prediction)
    return f"{str(np.round(prediction[0],2))} Lakh"


if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)