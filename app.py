from flask import Flask, render_template, request
import tensorflow as tf




app = Flask(__name__)


@app.route('/', methods=['get', 'post'])
def predict():
    message = ''
    if request.method == 'POST':
        iw_q = request.form.get('pole1')
        if_q = request.form.get('pole2')
        vw_q = request.form.get('pole3')
        fp_q = request.form.get('pole4')

        request_on_predict = [[float(iw_q), float(if_q), float(vw_q), float(fp_q), 8., 6.]]

        # pkl_ebw_mlp = "model\\ebw_mlp.txt"
        # with open(pkl_ebw_mlp, 'rb') as file:
        #     pickle_model = pickle.load(file)

        # model = joblib.load("model.joblib")

        model_loaded = tf.keras.models.load_model("model\\titanic_mlp")

        pred = model_loaded.predict(request_on_predict)
        message = f'Depth и Width {pred}'
        # message = f'Depth и Width {pred[0][1], pred[0][0]}'
        print(iw_q, if_q, vw_q, fp_q, 'predict', pred)

    return render_template('index.html', message=message)


# @app.route('/text/')  # Другая страница
# def print_text():
#     return "Some text!"


app.run()  # Закоментировать перед развёртыванием на сайте.
