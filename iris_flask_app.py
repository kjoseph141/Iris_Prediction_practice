from flask import Flask, render_template, request
from iris_predictor import get_flower_prediction

import os

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index_v4.html')

@app.route("/", methods = ['POST'])
def prediction():
    
    if request.method == 'GET':
        return render_template('index_v4.html')
    
    if request.method == 'POST':
    
        try:
            # Get input measurements from user
            
            sepal_len = request.form.get('sepal_len')
            sepal_wid = request.form.get('sepal_wid')
            petal_len = request.form.get('petal_len')
            petal_wid = request.form.get('petal_wid')
            
            #Generate prediction from input measurements
            
            species_prediction = get_flower_prediction(user_inputs=[sepal_len, sepal_wid, petal_len, petal_wid])
            
            if species_prediction == 'Iris Setosa':
                return render_template('result_setosa.html', pred = species_prediction)
            
            elif species_prediction == 'Iris Versicolor':
                return render_template('result_versicolor.html', pred = species_prediction)
            
            else:
                return render_template('result_virginica.html', pred = species_prediction)
        
        except:
            
            return render_template('result.html', pred = 'Try different values...')
            

          


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)