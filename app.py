from flask import Flask, render_template, request
import tensorflow as tf

import pickle
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__)


@app.route('/', methods=['get', 'post'])
def predict():
    message = ''
    if request.method == 'POST':
        iw_q = request.form.get('pole1')
        if_q = request.form.get('pole2')
        vw_q = request.form.get('pole3')
        fp_q = request.form.get('pole4')

        request_on_predict = [[float(iw_q), float(if_q), float(vw_q), float(fp_q)]]


        with open('model/ebw_mlp.pkl', 'rb') as file:
            loaded_model = pickle.load(file)


        pred = loaded_model.predict(request_on_predict)


        message = f'Depth и Width {pred[0][1], pred[0][0]}'

        print(iw_q, if_q, vw_q, fp_q, 'predict', pred)

    return render_template('index.html', message=message)



app.run()  # Закоментировать перед развёртыванием на сайте.
