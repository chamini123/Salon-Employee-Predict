from flask import Flask, request, render_template  
import pickle 
import numpy as np  
 
app = Flask(__name__)  
 
employee_model = pickle.load(open('model.pkl', 'rb'))
 
@app.route('/')  
def home():
    return render_template("home.html")
  
@app.route('/predict', methods=['POST'])  
def predict():
    int_features = [int(x) for x in request.form.values()]

    arr = [np.array(int_features)]
    pred = employee_model.predict(arr)  
    if pred == 0: 
      res_val = "no salary increment"  
    else:  
      res_val = "salary increment"  
    return render_template('home.html', prediction_text='The employee is likely to have a {}'.format(res_val))  
if __name__ == "__main__": 
    app.run()