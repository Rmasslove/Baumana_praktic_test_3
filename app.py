from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def print_hello():
    return render_template('index.html')


@app.route('/text/')
def print_text():
    return "Some text!"


app.run()
