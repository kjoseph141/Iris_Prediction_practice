from flask import Flask, render_template, request

from tensorflow.python.keras.backend import set_session
import keras
import tensorflow as tf
import numpy as np

import os

app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

# Load and set model
NN_model = keras.models.load_model('Iris_NN_model.h5')



@app.route('/')
def index():
    return render_template("index_v3.html")

@app.route("/", methods = ["POST"])
def prediction():
    
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
    
        sepal_len = request.form.get("Sepal Length")
        sepal_wid = request.form.get("Sepal Width")
        petal_len = request.form.get("Petal Length")
        petal_wid = request.form.get("Petal Width")
        
        #Generate prediction from input measurements
        species_list = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
    
        input_measures = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
        pred_on_input = (np.round(NN_model.predict(input_measures), decimals=0) != 0)[0]
        
        print(NN_model.predict(input_measures))
        print(np.round(NN_model.predict(input_measures), decimals=0))
        print((np.round(NN_model.predict(input_measures), decimals=0) != 0))
        print(pred_on_input)
        species_prediction = species_list[pred_on_input]
            
        return render_template('index_v3.html', 
                               sepal_len = sepal_len,
                               sepal_wid = sepal_wid,
                               petal_len = petal_len,
                               petal_wid = petal_wid,
                               pred = species_prediction[0])
          


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)