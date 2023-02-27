from flask import Flask, render_template, request
from flask_cors import CORS

import base64

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
        img = request.form.get('testing')

        FILE_WRITE = open("user_drawings/draw.txt", 'w')
        FILE_WRITE.write(img)
        FILE_WRITE.close()

        #TODO: Make decoder for IMG URL to IMG
        with open('sample.png', 'wb') as f:
            f.write(base64.decodebytes(img.split(',')[1].encode()))

        return "POST RECIEVED"


app.run(host = "0.0.0.0")