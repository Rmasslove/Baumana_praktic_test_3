from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


@app.route('/', methods=['get', 'post'])
def predict():

    message = ''

    if request.method == 'POST':
        iw_q = request.form.get('iw_q')
        if_q = request.form.get('if_q')
        vw_q = request.form.get('vw_q')
        fp_q = request.form.get('fp_q')

        request_on_predict = [[float(iw_q), float(if_q), float(vw_q), float(fp_q)]]

        with open('model/ebw_mlp.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('model/scaler_mlp.pkl', 'rb') as sc:
            scaler = pickle.load(sc)

        pr_sc = scaler.transform(request_on_predict)
        pred = loaded_model.predict(pr_sc)

        message = f'Depth и Width {pred[0][1], pred[0][0]}'

        print(iw_q, if_q, vw_q, fp_q, pred, pr_sc, scaler)

    return render_template('index.html', message=message)


app.run()  # Закоментировать перед развёртыванием на сайте.
