from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/draw')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def draw():

    if request.method == 'POST':
        print("Sent Back Data")
        

    else:
        return render_template('demoCanvas.html')

@app.route('/submit', methods=['GET','POST'])
def cheetos():
    if request.method =='POST':
        return 'I AM IN SUBMIT'

app.run(host = "0.0.0.0")